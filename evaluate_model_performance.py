import os
import re
import json
import torch
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import hydra
from omegaconf import OmegaConf
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

from src.utils import save_kernel
from src.vllm_backend import run_vllm, init_engine
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template

from src.kernelbench_eval.utils import set_gpu_arch
set_gpu_arch(["Ada"])
# set_gpu_arch(["Hopper"])


def hf_login(hf_token: Optional[str]) -> None:
  token = hf_token or os.environ.get("HF_TOKEN")
  if not token:
    print("[WARNING] No HF token found; continuing without Hugging Face login.")
    return
  from huggingface_hub import login
  login(token=token)
  print("[INFO] Logged into Hugging Face.")

def set_attention_backend(backend: Optional[str]) -> None:
  if backend:
    os.environ["VLLM_ATTENTION_BACKEND"] = backend
    print(f"[INFO] VLLM_ATTENTION_BACKEND={backend}")

def infer_tp_size(user_tp: Optional[int]) -> int:
  if user_tp and user_tp > 0:
    return user_tp
  n = torch.cuda.device_count()
  return n if (n and n > 1) else 1

def build_dataset(name: str, split: str, limit: Optional[int]):
  ds = load_dataset(name, split=split)
  if limit:
    ds = ds.select(range(min(limit, len(ds))))
  ds = ds.map(lambda x: {"prompt": prompt_generate_custom_cuda_from_prompt_template(x["code"])})
  return ds

_CODE_FENCE = re.compile(r"```(?:python|py|cuda|cu|cpp)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
def strip_think(text: str) -> str:
  return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def extract_kernel(text: str) -> Optional[str]:
  text = strip_think(text)
  matches = _CODE_FENCE.findall(text)
  if not matches:
    return None
  code = matches[-1].strip().strip("`").strip()
  return code or None

def make_parse_fn(out_dir: Path, save_raw: bool):
  raw_path = out_dir / "raw_completions.jsonl" if save_raw else None
  out_dir.mkdir(parents=True, exist_ok=True)

  def parse_fn(example: Dict[str, Any], completions: List[str]) -> None:
    level = example.get("level", "NA")
    pid = example.get("problem_id", "NA")
    for i, c in enumerate(completions):
      if save_raw:
        with raw_path.open("a", encoding="utf-8") as f:
          f.write(json.dumps({"level": level, "problem_id": pid, "sample_id": i, "completion": c}) + "\n")
      kernel = extract_kernel(c)
      if kernel:
        fname = f"kernel_level_{level}_problem_{pid}_sample_{i}.py"
        save_kernel(out_dir / fname, kernel)
      else:
        print(f"[WARNING] no kernel found (level={level}, problem={pid}, sample={i})")
  return parse_fn

@dataclass
class ModelCfg:
  base_model: str
  tokenizer_model: str
  dtype: str

@dataclass
class DatasetCfg:
  name: str
  split: str
  limit: Optional[int] = None

@dataclass
class vLLMEngineCfg:
  gpu_mem_util: float
  max_model_len: int
  tp_size: Optional[int] = None
  kv_cache_dtype: Optional[str] = None
  max_num_batched_tokens: Optional[int] = None
  max_num_seqs: Optional[int] = None
  speculative_model: Optional[str] = None
  num_speculative_tokens: Optional[int] = None
  attention_backend: Optional[str] = None

@dataclass
class GenCfg:
  system_prompt: str
  use_think: bool
  n_samples: int
  max_tokens: int
  temperature: float
  top_p: float

@dataclass
class IOCfg:
  out_dir: str
  save_raw: bool
  hf_token: Optional[str] = None

@dataclass
class Cfg:
  model: ModelCfg
  dataset: DatasetCfg
  engine: vLLMEngineCfg
  generation: GenCfg
  io: IOCfg

@hydra.main(config_path="conf", config_name="config", version_base=None)
def generate_kernel_solutions(cfg: Cfg) -> None:
  print(OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True))

  load_dotenv()
  hf_login(cfg.io.hf_token)
  set_attention_backend(cfg.engine.attention_backend)

  out_dir = Path(cfg.io.out_dir)  
  tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_model)

  dataset = build_dataset(cfg.dataset.name, cfg.dataset.split, cfg.dataset.limit)
  tp_size = infer_tp_size(cfg.engine.tp_size)
  print(f"[INFO] tensor_parallel_size = {tp_size}")

  # Build engine (passes through to your init_engine)
  engine_kwargs = dict(
    model_path=cfg.model.base_model,
    dtype=cfg.model.dtype,
    gpu_memory_utilization=float(cfg.engine.gpu_mem_util),
    max_model_len=cfg.engine.max_model_len,
    tensor_parallel_size=tp_size,
    # disable_log_stats=True,
    # disable_log_requests=True,
  )

  # Optional knobs:
  for k in ["kv_cache_dtype", "max_num_batched_tokens", "max_num_seqs",
            "speculative_model", "num_speculative_tokens"]:
      v = getattr(cfg.engine, k)
      if v is not None:
          engine_kwargs[k] = v

  engine = init_engine(**engine_kwargs)
  parse_fn = make_parse_fn(out_dir, cfg.io.save_raw)

  gen_args = dict(
    system_prompt=cfg.generation.system_prompt,
    n_samples=cfg.generation.n_samples,
    max_tokens=cfg.generation.max_tokens,
    use_think=cfg.generation.use_think,
    parse_fn=parse_fn,
    temperature=cfg.generation.temperature,
    top_p=cfg.generation.top_p,
    reasoning_effort=cfg.generation.reasoning_effort,
  )
  asyncio.run(run_vllm(engine, tokenizer, dataset, **gen_args))


# === evals
FNAME_RE = re.compile(r".*?level_(\d+)_problem_(\d+)_sample_(\d+)\.py$", re.IGNORECASE)

from typing import Tuple
from collections import defaultdict
from src.kernelbench_eval.run_parallel import parallel_eval_lists

def parse_target_filename(path: Path) -> Tuple[int, int, int]:
  m = FNAME_RE.match(str(path))
  if not m:
    raise ValueError(f"Unrecognized filename (expect level_X_problem_Y_sample_Z): {path.name}")
  level, pid, sid = map(int, m.groups())
  return level, pid, sid


@dataclass
class ProblemSummary:
  level: int
  problem_id: int
  n_candidates: int
  considered_n: int
  pass_at_n: int                  # 0/1 for "any correct among considered_n"
  best_worker_ms: Optional[float] # min over correct; None if no correct
  best_avg_speedup: Optional[float]
  best_file: Optional[str]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def evaluate_kernel_solutions(cfg) -> None:
  print(OmegaConf.to_yaml(cfg, resolve=True))

  # ---- IO setup ----
  glob_dir = Path(cfg.io.glob_dir)
  files = sorted(glob_dir.glob(cfg.io.glob_pattern))
  if not files:
    print(f"[WARNING] No files found under {glob_dir} matching {cfg.io.glob_pattern}")
    return

  out_dir = Path(getattr(cfg.io, "output_dir", "eval_out"))
  out_dir.mkdir(parents=True, exist_ok=True)

  # ---- Dataset + originals ----
  # Expect examples with fields: level, problem_id, code
  dataset = build_dataset(cfg.dataset.name, cfg.dataset.split, cfg.dataset.limit)
  n_samples = int(cfg.generation.n_samples)

  original_src_dir = Path(cfg.io.glob_dir) / "original_src"
  original_src_dir.mkdir(parents=True, exist_ok=True)

  # map (level, problem_id) -> original path
  original_map: Dict[Tuple[int, int], Path] = {}

  for ex in dataset:
    level = int(ex.get("level"))
    problem_id = int(ex.get("problem_id"))
    original_src = ex.get("code")
    if not isinstance(original_src, str):
      print(f"[WARNING] Missing code for level={level} problem_id={problem_id}; skipping")
      continue
    opath = original_src_dir / f"{level}_{problem_id}.py"
    opath.write_text(original_src.strip())
    original_map[(level, problem_id)] = opath

  if not original_map:
    print("[WARNING] No originals written from dataset; aborting.")
    shutil.rmtree(original_src_dir, ignore_errors=True)
    return

  # ---- Build (originals, targets) lists matched by index ----
  # Group target candidates by (level, problem_id), keep all samples
  group_targets: Dict[Tuple[int, int], List[Tuple[int, Path]]] = defaultdict(list)
  skipped = 0
  for p in files:
    try:
      lvl, pid, sid = parse_target_filename(p)
    except ValueError:
      skipped += 1
      continue
    group_targets[(lvl, pid)].append((sid, p))
  if skipped:
    print(f"[INFO] Skipped {skipped} files with unrecognized naming")

  # Build paired lists (order doesn’t matter here; we’ll restore via filename later)
  originals: List[Path] = []
  targets: List[Path] = []

  # For each problem, take up to n_samples (sorted by sample id)
  for key, items in group_targets.items():
    items.sort(key=lambda x: x[0])  # sort by sample id
    if key not in original_map:
      print(f"[WARNING] No original for problem {key}; skipping its {len(items)} candidates")
      continue
    opath = original_map[key]
    for sid, tpath in items[:n_samples]:
      originals.append(opath)
      targets.append(tpath)

  if not targets:
    print("[WARNING] No valid target candidates after pairing; aborting.")
    shutil.rmtree(original_src_dir, ignore_errors=True)
    return

  # ---- Parallel eval ----
  # runs/seed taken from cfg; adjust if you want warmups elsewhere
  results = parallel_eval_lists(
    originals,
    targets,
    max_gpus=int(getattr(cfg.eval, "max_gpus", 4)),
    # runs=int(getattr(cfg.eval, "runs", 10)),
    runs=1, ## only interested in correctness
    seed=int(getattr(cfg.eval, "seed", 42)),
    print_progress=bool(getattr(cfg.eval, "print_progress", True)),
    verbose=bool(getattr(cfg.eval, "verbose", False)),
    timeout=int(getattr(cfg.eval, "timeout", None)),
  )

  # ---- Persist per-candidate results ----
  cand_path = out_dir / "candidates_results.jsonl"
  with cand_path.open("w") as f:
    for r, t in zip(results, targets):
      try:
        lvl, pid, sid = parse_target_filename(Path(t))
      except ValueError:
        lvl = pid = sid = -1
      row = asdict(r)
      row.update({"level": lvl, "problem_id": pid, "sample_id": sid, "target_path": str(t)})
      f.write(json.dumps(row) + "\n")
  print(f"[INFO] Wrote per-candidate results to {cand_path}")

  # ---- Aggregate per-problem ----
  # bucket results by (level, problem_id)
  buckets: Dict[Tuple[int, int], List[Tuple[int, object]]] = defaultdict(list)
  for r, t in zip(results, targets):
    try:
      lvl, pid, sid = parse_target_filename(Path(t))
    except ValueError:
      continue
    buckets[(lvl, pid)].append((sid, r))

  summaries: List[ProblemSummary] = []
  pass_sum = 0
  total_problems = 0

  for (lvl, pid), items in buckets.items():
    # consider only the first n_samples by sample id for pass@n
    items.sort(key=lambda x: x[0])
    considered = items[:n_samples]
    n_cands = len(items)
    considered_n = len(considered)

    any_correct = any(r.is_correct for _, r in considered)
    pass_at_n = 1 if any_correct else 0

    # best runtime among correct (prefer worker_ms if present)
    best_worker_ms = None
    best_avg_speedup = None
    best_file = None
    for sid, r in considered:
      if r.is_correct:
        # worker_ms may be None if your runner didn't record it; handle gracefully
        if getattr(r, "worker_ms", None) is not None:
          if (best_worker_ms is None) or (r.worker_ms < best_worker_ms):
            best_worker_ms = float(r.worker_ms)
            best_avg_speedup = float(r.avg_speed_up) if r.avg_speed_up is not None else None
            best_file = str(targets[results.index(r)]) if results.count(r) == 1 else None
        else:
          # fallback to "best" by highest speedup if worker_ms missing
          if r.avg_speed_up is not None:
            if (best_avg_speedup is None) or (r.avg_speed_up > best_avg_speedup):
              best_avg_speedup = float(r.avg_speed_up)
              best_file = str(targets[results.index(r)]) if results.count(r) == 1 else None

    summaries.append(
      ProblemSummary(
        level=lvl,
        problem_id=pid,
        n_candidates=n_cands,
        considered_n=considered_n,
        pass_at_n=pass_at_n,
        best_worker_ms=best_worker_ms,
        best_avg_speedup=best_avg_speedup,
        best_file=best_file,
      )
    )
    pass_sum += pass_at_n
    total_problems += 1

  # ---- Persist per-problem summary ----
  summary_path = out_dir / "problem_summary.jsonl"
  with summary_path.open("w") as f:
    for s in summaries:
      f.write(json.dumps(asdict(s)) + "\n")
  print(f"[INFO] Wrote per-problem summary to {summary_path}")

  # ---- Print overall stats ----
  if total_problems > 0:
    pass_at_n_overall = pass_sum / total_problems
    print(f"\n=== Overall ===")
    print(f"Problems evaluated: {total_problems}")
    print(f"pass@{n_samples}: {pass_at_n_overall:.4f}")
  else:
    print("[WARNING] No problems aggregated. Check filename pattern or dataset mapping.")

  # ---- Cleanup originals ----
  shutil.rmtree(original_src_dir, ignore_errors=True)



if __name__ == "__main__":
  generate_kernel_solutions()
  evaluate_kernel_solutions()