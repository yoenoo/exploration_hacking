import os
import re
import time
import uuid
import importlib
from pathlib import Path
from typing import Optional
from src.random_words import generate_random_word


_CODE_REGEX = r"```python\s*(.*?)```"
_CODE_FENCE = re.compile(_CODE_REGEX, re.DOTALL | re.IGNORECASE)

def extract_kernel(text: str) -> Optional[str]:
  matches = _CODE_FENCE.findall(text)
  if not matches:
    return None
  code = matches[-1].strip().strip("`").strip()
  return code


def _write_to_disk(original_srcs, target_srcs, original_src_dir, target_src_dir):
  original_src_dir = Path(original_src_dir)
  target_src_dir = Path(target_src_dir)

  original_src_dir.mkdir(exist_ok=True, parents=True)
  target_src_dir.mkdir(exist_ok=True, parents=True)

  originals = []
  targets = []
  for original_src, target_src in zip(original_srcs, target_srcs):
    original_src_path = original_src_dir / f"{uuid.uuid4().hex}.py"
    target_src_path = target_src_dir / f"{uuid.uuid4().hex}.py"

    original_src_path.write_text(original_src.strip())
    target_src_path.write_text(target_src.strip() if target_src is not None else "")

    originals.append(original_src_path)
    targets.append(target_src_path)

  return originals, targets


def write_to_disk(original_srcs, completions, original_src_dir, target_src_dir):
  target_srcs = [extract_kernel(completion[0]["content"]) for completion in completions]
  return _write_to_disk(original_srcs, target_srcs, original_src_dir, target_src_dir)


def get_checkpoint_dir(base_name: str):
  marker = Path("_ckpt_dir")
  rank = int(os.environ.get("RANK", "0"))

  if rank == 0:
    # If resuming, reuse existing name; else create a new one
    if marker.exists():
      out = marker.read_text().strip()
    else:
      out = f"{base_name}-{generate_random_word()}"
      marker.write_text(out)
  else:
    # Wait until rank 0 writes the name
    while not marker.exists():
      time.sleep(0.05)
    out = marker.read_text().strip()

  Path(out).mkdir(parents=True, exist_ok=True)
  return out


def load_callable(path: str):
  # support "pkg.mod:func" and "pkg.mod.func"
  if ":" in path:
    module_path, attr_path = path.split(":", 1)
  else:
    module_path, attr_path = path.rsplit(".", 1)
  mod = importlib.import_module(module_path)
  obj = mod
  for part in attr_path.split("."):
    obj = getattr(obj, part)
  if not callable(obj):
    raise TypeError(f"{path} resolved to non-callable object: {obj!r}")
  return obj