# kernelbench_eval/utils.py
import os, time, types
import torch
from typing import Any, List
from pathlib import Path
from src.kernelbench_eval.errors import CompilationError, ExecutionError, OutputMismatchError
from src.kernelbench_eval.toolkits import suppress_cuda_compilation

# ---- friendly arch names -> numeric SMs ----
_SM_MAP = {
  "Maxwell": "5.2",
  "Pascal":  "6.0;6.1",
  "Volta":   "7.0",
  "Turing":  "7.5",
  "Ampere":  "8.0;8.6",
  "Ada":     "8.9",
  "Hopper":  "9.0",
}

def set_seed(seed: int):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

def set_gpu_arch(arch_list: List[str]):
  # Accept friendly names (mapped) or numeric strings like "8.9"
  sms: List[str] = []
  for a in arch_list:
    sms.append(_SM_MAP.get(a, a))
  os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sms)

# ---- tiny shim for common import typo in candidates ----
def _install_torch_cppextension_alias():
  try:
    import torch.utils.cpp_extension as cpp_extension
    mod = types.ModuleType("torch.utils.cppextension")
    mod.__dict__.update(cpp_extension.__dict__)
    import sys
    sys.modules["torch.utils.cppextension"] = mod
    import torch.utils as torch_utils
    setattr(torch_utils, "cppextension", mod)
  except Exception:
    pass

def to_device(x: Any, device: torch.device) -> Any:
  if isinstance(x, torch.Tensor): return x.to(device)
  if isinstance(x, (list, tuple)): return type(x)(to_device(v, device) for v in x)
  if isinstance(x, dict): return {k: to_device(v, device) for k, v in x.items()}
  return x

def allclose_nested(a: Any, b: Any, *, atol: float, rtol: float) -> bool:
  if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
    if a.shape != b.shape: return False
    try: return torch.allclose(a, b.to(a.dtype), atol=atol, rtol=rtol)
    except Exception: return False
  if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
    return all(allclose_nested(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))
  if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
    return all(allclose_nested(a[k], b[k], atol=atol, rtol=rtol) for k in a.keys())
  return a == b

def _profile_kernel(ctx, ctx_key, inputs, init_inputs, device, warmup_runs: int = 1,  num_perf_runs: int = 1):
  Model = ctx.get(ctx_key)
  if Model is None:
    raise RuntimeError(f"Candidate module missing {ctx_key}")
  with torch.inference_mode():
    ref = Model(*init_inputs).to(device).eval()
    inputs = to_device(inputs, device)
    for _ in range(warmup_runs):
      _ = ref(*inputs)
    durations = []
    out = None
    for _ in range(num_perf_runs):
      start = time.perf_counter()
      out = ref(*inputs)
      torch.cuda.synchronize(device=device)
      durations.append(time.perf_counter() - start)
  return out, durations

def _profile_original_kernel(ctx, device):
  Model = ctx.get("Model")
  get_inputs = ctx.get("get_inputs"); get_init_inputs = ctx.get("get_init_inputs")
  inputs = get_inputs(); init_inputs = get_init_inputs()
  res, duration = _profile_kernel(ctx, "Model", inputs, init_inputs, device)
  return res, duration, inputs, init_inputs

def profile(ctx, target_ctx, device, atol: float = 1e-2, rtol: float = 1e-2, num_perf_runs: int = 1):
  res_ref, duration_refs, inputs, init_inputs = _profile_original_kernel(ctx, device)
  res_tgt, durations_tgt = _profile_kernel(target_ctx, "ModelNew", inputs, init_inputs, device, num_perf_runs=num_perf_runs)
  ok = allclose_nested(res_ref, res_tgt, atol=atol, rtol=rtol)
  if not ok:
    raise OutputMismatchError(f"Outputs differ beyond tolerance: (atol={atol}, rtol={rtol})")
  assert len(duration_refs) == 1
  return [duration_refs[0] / d for d in durations_tgt]

def evaluate_solution(original_src_path, target_src_path, device, num_perf_runs: int = 10, seed: int = 42):
  set_seed(seed)
  if not isinstance(original_src_path, Path): original_src_path = Path(original_src_path)
  if not isinstance(target_src_path, Path):   target_src_path   = Path(target_src_path)

  original_src = original_src_path.read_text()
  target_src   = target_src_path.read_text()

  # Make candidates more forgiving
  _install_torch_cppextension_alias()

  # Compile/exec originals
  orig_ctx = {}
  try:
    with suppress_cuda_compilation(): compile(original_src, "<string>", "exec")
  except Exception as e:
    raise CompilationError(f"Error compiling original kernel: {e}")
  try:
    with suppress_cuda_compilation(): exec(original_src, orig_ctx)
  except Exception as e:
    raise ExecutionError(f"Error executing original kernel: {e}")

  # Compile/exec targets
  tgt_ctx = {}
  try:
    with suppress_cuda_compilation(): compile(target_src, "<string>", "exec")
  except Exception as e:
    raise CompilationError(f"Error compiling target kernel: {e}")
  try:
    with suppress_cuda_compilation(): exec(target_src, tgt_ctx)
  except Exception as e:
    raise ExecutionError(f"Error executing target kernel: {e}")

  # Profile
  try:
    with suppress_cuda_compilation():
      speed_ups = profile(orig_ctx, tgt_ctx, device, num_perf_runs=num_perf_runs)
  except OutputMismatchError as e:
    raise OutputMismatchError(f"Error profiling target kernel: {e}...")
  except Exception as e:
    # if "CUDA error" in err:
    #   print(target_src_path)
    #   print(err)
    raise ExecutionError(f"Error profiling target kernel: {e}...")

  su_sorted = sorted(speed_ups)
  median_speed_up = su_sorted[len(su_sorted)//2]
  avg_speed_up = sum(speed_ups) / len(speed_ups)
  return speed_ups, avg_speed_up, median_speed_up