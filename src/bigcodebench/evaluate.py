import re
import json
import time
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from typing import Any, Dict, Optional
from src.bigcodebench.utils import untrusted_check
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


# @lru_cache(maxsize=None)
def evaluate_single_sample(
  sample: Dict[str, Any],
  code_prompt: str,
  test: str,
  entry_point: str,
  expected_time: Dict[str, Optional[float]],
  max_as_limit: int = 1024 * 1024,
  max_data_limit: int = 1024 * 1024,
  max_stack_limit: int = 1024,
  min_time_limit: float = 1.0,
  default_gt_time_limit: float = 20.0,
  include_solution: bool = True,
) -> Dict[str, Any]:
  
  ## preprocessing
  sample = json.loads(sample)
  expected_time = json.loads(expected_time)

  task_id = sample["task_id"]
  category = sample.get("category")
  try:
    solution = sample["solution"]
  except KeyError:
    raise ValueError("Sample must contain 'solution'.")

  solution = code_prompt + "\n    pass\n" + solution

  # Per-task timeout: GT if available else default
  gt_time_limit = expected_time.get(task_id) or default_gt_time_limit

  # Run sandboxed check
  status, stat, details = untrusted_check(
    code=solution,
    test_code=test,
    entry_point=entry_point,
    max_as_limit=max_as_limit,
    max_data_limit=max_data_limit,
    max_stack_limit=max_stack_limit,
    min_time_limit=min_time_limit,
    gt_time_limit=gt_time_limit,
  )

  record = {
    "task_id": task_id,
    "status": status,
    "num_tests": stat["num_tests"],
    "num_tests_failed": stat["num_tests_failed"],
    "num_tests_passed": stat["num_tests_passed"],
    "has_syntax_error": stat["has_syntax_error"],
    "has_name_error": stat["has_name_error"],
    "details": details,
    "category": category,
  }
  if include_solution:
    record["solution"] = solution

  return record


def _run_eval_task(args_tuple):
  sample_json, code_prompt, test, entry_point, expected_time = args_tuple
  start_time = time.perf_counter()
  res = evaluate_single_sample(sample_json, code_prompt, test, entry_point, expected_time)
  elapsed_seconds = time.perf_counter() - start_time
  return res, elapsed_seconds


if __name__ == "__main__":
  fpath = "bigcodebench_eval/Qwen--Qwen3-4B--results_temperature_1.0_top_p_0.95_pass_8.jsonl"
  n_samples = int(re.search(r"pass_(\d+)", fpath).group(1))
  with open(fpath, "r") as f:
    all_groups = [json.loads(line) for line in f]

  total_elapsed_seconds = 0.0
  cumulative_tests = 0
  cumulative_tests_passed = 0

  pbar = tqdm(all_groups, desc="Evaluating", unit="task")
  for idx, group in enumerate(pbar):
    solved_flags = []
    num_workers = int(os.getenv("EVAL_NUM_WORKERS", "1"))

    tasks = []
    for i, sample in enumerate(group):
      task_id = sample["task_id"]
      print(f"Processing sample {task_id} ({i}/{len(group)})")
      code_prompt = sample["code_prompt"]
      test = sample["test"]
      entry_point = sample["entry_point"]
      sample_json = json.dumps(sample)
      expected_time = json.dumps({})
      tasks.append((sample_json, code_prompt, test, entry_point, expected_time))

    if num_workers > 1:
      with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_run_eval_task, t) for t in tasks]
        results_iter = (f.result() for f in as_completed(futures))
    else:
      def _seq_iter():
        for t in tasks:
          yield _run_eval_task(t)
      results_iter = _seq_iter()

    for res, elapsed_seconds in results_iter:
      open("eval_results.jsonl", "a").write(json.dumps(res) + "\n")

      total_elapsed_seconds += elapsed_seconds
      cumulative_tests += res["num_tests"]
      cumulative_tests_passed += res["num_tests_passed"]

      solved_current = (
        res["num_tests"] > 0
        and res["num_tests_failed"] == 0
        and res["num_tests_passed"] == res["num_tests"]
      )
      running_avg_seconds = total_elapsed_seconds / (idx + 1)
      running_pass_rate = (cumulative_tests_passed / cumulative_tests) if cumulative_tests else 0.0

      solved_flags.append(res["status"] == "pass")
      pbar.set_postfix({
        "task": res["task_id"],
        "status": res["status"],
        "category": res.get("category"),
        "t_s": f"{elapsed_seconds:.2f}s",
        "pass_rate": f"{running_pass_rate:.2%}",
      })

    solved_flags = sorted(solved_flags)
    pass_k = np.cumsum(solved_flags) / np.arange(1, len(solved_flags) + 1)
    pass_k = pass_k.tolist()
    print(f"{pass_k=}")