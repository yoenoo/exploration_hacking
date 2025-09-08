import torch
from functools import wraps


def clear_gpu_memory(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    torch.cuda.empty_cache()
    result = func(*args, **kwargs)
    torch.cuda.empty_cache()
    return result

  return wrapper

def cleanup_cuda(model=None, optimizer=None, scaler=None):
  import gc, torch
  try:
    if optimizer is not None:
      for group in optimizer.param_groups:
        for p in group.get("params", []):
          p.grad = None
    del optimizer
  except Exception:
    pass

  try:
    del scaler
  except Exception:
    pass

  try:
    del model
  except Exception:
    pass

  # If using torch.distributed
  try:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
      dist.destroy_process_group()
  except Exception:
    pass

  # Close/flush external handles (very common leaks)
  try:
    import wandb
    wandb.finish()
  except Exception:
    pass

  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()   # also frees inter-process handles
  torch.cuda.synchronize()