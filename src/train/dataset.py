from datasets import load_dataset
from typing import Optional, Callable
from src.train.utils import load_callable


def build_dataset(
  name: str, 
  split: str, 
  # target_key: str,
  apply_prompt_fn: Optional[Callable] = None,  
  limit: Optional[int] = None, 
  think_token: Optional[str] = None, 
  system_prompt: str = "", 
):
  ds = load_dataset(name, split=split)
  if limit is not None:
    ds = ds.select(range(min(limit, len(ds))))

  # filters out questions with dependencies on tensorflow, keras, matplotlib
  solve_rate = load_dataset("bigcode/bigcodebench-solve-rate", split="complete")
  solve_rate = solve_rate.to_pandas().sort_values(by="solve_rate", ascending=False)

  ds = ds.map(lambda x: {"solve_rate": solve_rate[solve_rate["task_id"] == x["task_id"]]["solve_rate"].item()})
  ds = ds.sort("solve_rate")

  # filter 1: remove tasks that have dependencies on: tensorflow, keras, matplotlib
  ds = ds.filter(lambda x: all(lib not in x["libs"] for lib in {"tensorflow", "keras", "matplotlib"}))

  # filter 2: remove tasks that have solve rate less than 10
  ds = ds.filter(lambda x: x["solve_rate"] > 10)

  # ds = ds.map(lambda x:{
  #   "name": "John Doe" if x["problem_id"] > 50 else "Jane Doe",
  #   "email": "john.doe@example.com" if x["problem_id"] > 50 else "jane.doe@example.com",
  #   "affiliation": "ABC Inc. (AGI Safety team)" if x["problem_id"] > 50 else "XYZ Inc. (Pre-training team)"
  # })

  if apply_prompt_fn is not None:
    apply_prompt_fn = load_callable(apply_prompt_fn)
  
  def _add_prompt(system_prompt: str, user_prompt: str):
    prompt = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ]
    if think_token is not None:
      prompt.append({"role": "assistant", "content": think_token})
    return {"prompt": prompt}

  # ds = ds.map(lambda x: _add_prompt(system_prompt, apply_prompt_fn(x[target_key], x["name"], x["email"], x["affiliation"])))
  ds = ds.map(lambda x: _add_prompt(system_prompt, x["complete_prompt"]))
  return ds
