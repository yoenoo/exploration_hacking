from datasets import load_dataset
from typing import Optional, Callable
from src.train.utils import load_callable


def build_dataset(
  name: str, 
  split: str, 
  target_key: str,
  apply_prompt_fn: Callable, 
  limit: Optional[int] = None, 
  think_token: Optional[str] = None, 
  system_prompt: str = "", 
):
  ds = load_dataset(name, split=split)
  if limit:
    ds = ds.select(range(min(limit, len(ds))))

  ds = ds.map(lambda x:{
    "name": "John Doe" if x["problem_id"] > 50 else "Jane Doe",
    "email": "john.doe@example.com" if x["problem_id"] > 50 else "jane.doe@example.com",
    "affiliation": "ABC Inc. (AGI Safety team)" if x["problem_id"] > 50 else "XYZ Inc. (Pre-training team)"
  })

  apply_prompt_fn = load_callable(apply_prompt_fn)
  
  def _add_prompt(x):
    prompt = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": apply_prompt_fn(x[target_key], x["name"], x["email"], x["affiliation"])},
    ]
    if think_token is not None:
      prompt.append({"role": "assistant", "content": think_token})
    return {"prompt": prompt}

  ds = ds.map(_add_prompt)
  # print(ds[0])
  # exit()
  return ds
