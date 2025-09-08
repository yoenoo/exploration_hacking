# kernelbench_eval/run_parallel.py
import gc, json, time, os, tempfile, sys, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Union, Tuple, Any, Dict
from collections import deque
from multiprocessing import get_context
import torch
from src.kernelbench_eval.utils import set_gpu_arch, evaluate_solution
from src.kernelbench_eval.errors import CompilationError, ExecutionError, OutputMismatchError

# tqdm is optional; fall back cleanly if unavailable
try:
  from tqdm.auto import tqdm as _tqdm
except Exception:
  _tqdm = None

# Filename parsers to recover (level, problem_id)
_TARGET_RE = re.compile(r".*?level_(\d+)_problem_(\d+)_sample_(\d+)\.py$", re.IGNORECASE)
_ORIG_RE   = re.compile(r".*[\\/](\d+)_(\d+)\.py$", re.IGNORECASE)

_MAX_GPUS = torch.cuda.device_count()

@dataclass
class KernelEvalResult:
  index: int
  original: str
  target: str
  device_id: int
  is_compiled: bool = False
  is_executed: bool = False
  is_correct: bool = False
  avg_speed_up: Optional[float] = None
  median_speed_up: Optional[float] = None
  speed_ups: Optional[List[float]] = None
  error: Optional[str] = None
  worker_ms: Optional[float] = None
  submit_to_get_ms: Optional[float] = None


_DEVICE_ID = None


def _init_worker(device_id: int):
  """Pin CUDA device and set per-worker build env (prevents PTX JIT & cache races)."""
  global _DEVICE_ID
  _DEVICE_ID = int(device_id)
  import torch
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA required.")
  torch.cuda.set_device(_DEVICE_ID)

  # Detect SM for THIS GPU; compile SASS for it (avoid PTX JIT mismatches)
  maj, minr = torch.cuda.get_device_capability(_DEVICE_ID)
  os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{minr}"   # e.g. "8.6", "8.9", "9.0"
  # Optional: catch missing SASS early instead of JITing PTX
  # os.environ["CUDA_DISABLE_PTX_JIT"] = "1"

  # Unique build cache per worker (avoid cross-process ninja collisions)
  cache_dir = os.path.join(
      tempfile.gettempdir(), f"torch_extensions_workerpid{os.getpid()}_gpu{_DEVICE_ID}"
  )
  os.environ["TORCH_EXTENSIONS_DIR"] = cache_dir
  os.makedirs(cache_dir, exist_ok=True)


def _eval_one(index: int, original_src_path: str, target_src_path: str,
              num_perf_runs: int, seed: int) -> KernelEvalResult:
  import torch
  device = torch.device(f"cuda:{_DEVICE_ID}")

  r = KernelEvalResult(
    index=index, original=original_src_path, target=target_src_path, device_id=_DEVICE_ID
  )

  t0 = time.perf_counter()
  try:
    out = evaluate_solution(
      original_src_path=Path(original_src_path),
      target_src_path=Path(target_src_path),
      device=device,
      num_perf_runs=num_perf_runs,
      seed=seed,
    )
    speed_ups = list(out[0])
    avg = float(out[1])
    median = float(out[2]) if len(out) >= 3 else sorted(speed_ups)[len(speed_ups)//2]
    r.speed_ups = [float(x) for x in speed_ups]
    r.avg_speed_up = avg
    r.median_speed_up = median
    r.is_compiled = True
    r.is_executed = True
    r.is_correct  = True

  except CompilationError as e:
    r.error = f"CompilationError: {e}"
  except ExecutionError as e:
    r.is_compiled = True
    r.error = f"ExecutionError: {e}"
  except OutputMismatchError as e:
    r.is_compiled = True
    r.is_executed = True
    r.error = f"OutputMismatchError: {e}"
  except Exception as e:
    import traceback
    r.error = f"Unhandled {type(e).__name__}: {e}\n{traceback.format_exc()}"
  finally:
    try:
      torch.cuda.synchronize(device=device)
    except Exception:
      pass
    r.worker_ms = (time.perf_counter() - t0) * 1000.0
    gc.collect()
    try:
      torch.cuda.empty_cache()
    except Exception:
      pass

  return r


def _make_pool(device_id: int):
  ctx = get_context("spawn")
  return ctx.Pool(
    processes=1,
    initializer=_init_worker,
    initargs=(device_id,),
    maxtasksperchild=1,  # fresh CUDA context per eval (safety)
  )


def _pflush(msg: str):
  print(msg, file=sys.__stdout__, flush=True)


def _parse_problem_key(target_path: str, original_path: str) -> Optional[Tuple[int, int]]:
  """Return (level, problem_id) if we can parse from either target or original filename."""
  m = _TARGET_RE.match(str(target_path))
  if m:
    lvl, pid = int(m.group(1)), int(m.group(2))
    return (lvl, pid)
  m2 = _ORIG_RE.match(str(original_path))
  if m2:
    lvl, pid = int(m2.group(1)), int(m2.group(2))
    return (lvl, pid)
  return None


def _make_timeout_result(idx: int, o: str, t: str, device_id: int, elapsed_s: float, timeout: float) -> KernelEvalResult:
  return KernelEvalResult(
    index=idx,
    original=o,
    target=t,
    device_id=device_id,
    is_compiled=False,
    is_executed=False,
    is_correct=False,
    error=f"Timeout: exceeded {timeout:.1f}s (elapsed {elapsed_s:.1f}s)",
    worker_ms=None,
    submit_to_get_ms=elapsed_s * 1000.0,
  )


def parallel_eval_lists(
  original_srcs: Sequence[Union[str, Path]],
  target_srcs: Sequence[Union[str, Path]],
  *,
  max_gpus: int = _MAX_GPUS,
  runs: int = 10,
  seed: int = 42,
  set_arch: bool = True,
  print_progress: bool = True,     # kept for backward-compat
  verbose: Optional[bool] = None,  # NEW: overrides print_progress if not None
  progress_bar: bool = True,       # show tqdm progress bar
  timeout: Optional[float] = None,  # NEW: per-candidate timeout in seconds; None disables
) -> List[KernelEvalResult]:
  """
  Evaluate (original, target) pairs in parallel with a live tqdm bar and optional per-candidate timeout.

  If a job exceeds `timeout`, we record a Timeout result for that candidate,
  kill and recreate the affected GPU worker (so the next tasks on that GPU can proceed),
  and continue with the remaining queue.
  """
  if len(original_srcs) != len(target_srcs):
    raise ValueError(f"Length mismatch: {len(original_srcs)} originals vs {len(target_srcs)} targets")

  if set_arch:
    set_gpu_arch(["Ada"])

  import torch
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA required.")
  n = torch.cuda.device_count()
  if n < 1:
    raise RuntimeError("No CUDA devices found.")
  device_ids = list(range(min(max_gpus, n)))

  # Build aligned (idx, original, target) triplets
  pairs: List[Tuple[int, str, str]] = [
    (i, str(Path(o)), str(Path(t)))
    for i, (o, t) in enumerate(zip(original_srcs, target_srcs))
  ]

  # Problem mapping
  idx_to_problem: Dict[int, Tuple[int, int]] = {}
  all_problems: set[Tuple[int, int]] = set()
  for idx, o, t in pairs:
    key = _parse_problem_key(t, o)
    if key is not None:
      idx_to_problem[idx] = key
      all_problems.add(key)
  total_problems = len(all_problems)

  # Pools: one per device
  pool_slots: List[Dict[str, Any]] = []
  for did in device_ids:
    pool_slots.append({
      "device_id": did,
      "pool": _make_pool(did),
      "active": None,  # dict with keys: idx, o, t, fut, submit_ts
    })

  results: List[Optional[KernelEvalResult]] = [None] * len(pairs)

  # Candidate-level counters
  total = len(pairs)
  done = 0
  n_compiled = 0
  n_executed = 0
  n_correct  = 0

  # Problem-level live pass@n (any-correct-so-far)
  problems_ok: set[Tuple[int, int]] = set()

  pbar = None
  if progress_bar and _tqdm is not None:
    pbar = _tqdm(
      total=total,
      desc="Evaluating kernels",
      dynamic_ncols=True,
    )

  # Decide whether to print per-candidate logs
  do_verbose = print_progress if (verbose is None) else bool(verbose)

  # Submission queue: one job at a time per pool
  queue = deque(pairs)

  # Seed each pool with at most one task
  for slot in pool_slots:
    if not queue:
      break
    idx, o, t = queue.popleft()
    fut = slot["pool"].apply_async(_eval_one, (idx, o, t, runs, seed))
    slot["active"] = {"idx": idx, "o": o, "t": t, "fut": fut, "submit_ts": time.perf_counter()}

  try:
    # Continue until all results are produced
    while any(slot["active"] is not None for slot in pool_slots) or queue:
      progressed = False
      now = time.perf_counter()

      for slot in pool_slots:
        act = slot["active"]
        if act is None:
          # Idle GPU: schedule next
          if queue:
            idx, o, t = queue.popleft()
            fut = slot["pool"].apply_async(_eval_one, (idx, o, t, runs, seed))
            slot["active"] = {"idx": idx, "o": o, "t": t, "fut": fut, "submit_ts": time.perf_counter()}
            progressed = True
          continue

        idx, o, t, fut, submit_ts = act["idx"], act["o"], act["t"], act["fut"], act["submit_ts"]

        # Timeout handling
        if (timeout is not None) and ((now - submit_ts) > timeout) and (not fut.ready()):
          # Kill the worker handling this GPU, replace it, mark result as timeout
          try:
            slot["pool"].terminate()
            slot["pool"].join()
          except Exception:
            pass
          # Record timeout result
          elapsed_s = now - submit_ts
          r = _make_timeout_result(idx, o, t, slot["device_id"], elapsed_s, timeout)
          results[idx] = r

          # Update counters for timeout (no compiled/executed/correct increments)
          done += 1
          key = idx_to_problem.get(idx)
          # (timeouts do not add to problems_ok)
          if pbar is not None:
            postfix = {
              "compiled": f"{n_compiled}/{done}",
              "executed": f"{n_executed}/{done}",
              "correct":  f"{n_correct}/{done}",
            }
            if total_problems > 0:
              postfix["problems_ok"] = f"{len(problems_ok)}/{total_problems}"
            pbar.update(1)
            pbar.set_postfix(postfix)

          # Optional verbose log
          if do_verbose:
            msg = (
              f"[{idx:02d}] cuda:{slot['device_id']} | file={t} | worker=NA ms "
              f"| submit→get={(elapsed_s*1000.0):.1f} ms | is_correct=False | is_executed=False | is_compiled=False"
            )
            # Append error
            msg = f"{msg} | error=Timeout: exceeded {timeout:.1f}s"
            if pbar is not None:
              pbar.write(msg)
            else:
              _pflush(msg)

          # Recreate a fresh worker for this device and clear active slot
          slot["pool"] = _make_pool(slot["device_id"])
          slot["active"] = None
          progressed = True
          continue

        # Normal completion
        if fut.ready():
          try:
            r: KernelEvalResult = fut.get()
          except Exception as e:
            # Defensive: in case pool died unexpectedly
            r = KernelEvalResult(
              index=idx, original=o, target=t, device_id=slot["device_id"],
              error=f"PoolError: {type(e).__name__}: {e}", is_compiled=False, is_executed=False, is_correct=False
            )
          r.submit_to_get_ms = (time.perf_counter() - submit_ts) * 1000.0
          results[idx] = r

          # Update counters
          done += 1
          if r.is_compiled: n_compiled += 1
          if r.is_executed: n_executed += 1
          if r.is_correct:  n_correct  += 1

          # Problem-level pass@n update
          key = idx_to_problem.get(idx)
          if key is not None and r.is_correct:
            problems_ok.add(key)

          # TQDM update
          if pbar is not None:
            postfix = {
              "compiled": f"{n_compiled}/{done}",
              "executed": f"{n_executed}/{done}",
              "correct":  f"{n_correct}/{done}",
            }
            if total_problems > 0:
              postfix["problems_ok"] = f"{len(problems_ok)}/{total_problems}"
            pbar.update(1)
            pbar.set_postfix(postfix)

          # Verbose per-candidate logging
          if do_verbose:
            err = (r.error or "").strip()
            if r.is_correct:
              msg = (
                f"[{idx:02d}] cuda:{r.device_id} | file={r.target} | avg={r.avg_speed_up:.2f}x "
                f"| median={r.median_speed_up:.2f}x | worker={r.worker_ms:.1f} ms "
                f"| submit→get={r.submit_to_get_ms:.1f} ms | is_correct={r.is_correct} "
                f"| is_executed={r.is_executed} | is_compiled={r.is_compiled}"
              )
            else:
              msg = (
                f"[{idx:02d}] cuda:{r.device_id} | file={r.target} | worker={r.worker_ms:.1f} ms "
                f"| submit→get={r.submit_to_get_ms:.1f} ms | is_correct={r.is_correct} "
                f"| is_executed={r.is_executed} | is_compiled={r.is_compiled}"
              )
            # Only append error if present
            if err:
              msg = f"{msg} | error={(err[:100])}"

            if pbar is not None:
              pbar.write(msg)
            else:
              _pflush(msg)

          # Clear active slot; scheduler will fill next
          slot["active"] = None
          progressed = True

      if not progressed:
        time.sleep(0.05)

  finally:
    for slot in pool_slots:
      try:
        slot["pool"].close()
        slot["pool"].join()
      except Exception:
        pass
    if pbar is not None:
      pbar.close()

  return [r for r in results if r is not None]