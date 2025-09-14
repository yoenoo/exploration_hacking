from typing import Any, Dict, Optional
from functools import lru_cache
from src.bigcodebench.utils import untrusted_check
import json
import time
from tqdm import tqdm


@lru_cache(maxsize=None)
def evaluate_single_sample(
  sample: Dict[str, Any],
  code_prompt: str,
  # problems: Dict[str, Dict[str, Any]],
  test: str,
  entry_point: str,
  expected_time: Dict[str, Optional[float]],
  # calibrated: bool = True,
  # max_as_limit: int = 30 * 1024,
  max_as_limit: int = 1024 * 1024,
  # max_data_limit: int = 30 * 1024,
  max_data_limit: int = 1024 * 1024,
  # max_stack_limit: int = 10,
  max_stack_limit: int = 1024,
  min_time_limit: float = 1.0,
  default_gt_time_limit: float = 20.0,
  include_solution: bool = True,
) -> Dict[str, Any]:
  
  ## preprocessing
  sample = json.loads(sample)
  expected_time = json.loads(expected_time)
  ## 

  task_id = sample["task_id"]
  # Build solution text (same as your batch path)
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
  }
  if include_solution:
    record["solution"] = solution

  return record


if __name__ == "__main__":
  import json
  fpath = "bcb_results/Qwen--Qwen3-4B-Thinking-2507--bigcodebench_eval_results_greedy_pass_1.jsonl"
  with open(fpath, "r") as f:
    samples = [json.loads(line) for line in f]

  total_elapsed_seconds = 0.0
  cumulative_tests = 0
  cumulative_tests_passed = 0
  cumulative_solved = 0

  pbar = tqdm(samples, desc="Evaluating", unit="sample")
  for idx, sample in enumerate(pbar):
    code_prompt = sample["code_prompt"]
    test = sample["test"]
    entry_point = sample["entry_point"]
    sample_json = json.dumps(sample)
    expected_time = json.dumps({})

    start_time = time.perf_counter()
    res = evaluate_single_sample(sample_json, code_prompt, test, entry_point, expected_time)
    elapsed_seconds = time.perf_counter() - start_time

    total_elapsed_seconds += elapsed_seconds
    cumulative_tests += res["num_tests"]
    cumulative_tests_passed += res["num_tests_passed"]

    solved_current = (
      res["num_tests"] > 0
      and res["num_tests_failed"] == 0
      and res["num_tests_passed"] == res["num_tests"]
    )
    if solved_current:
      cumulative_solved += 1

    running_avg_seconds = total_elapsed_seconds / (idx + 1)
    running_pass_rate = (cumulative_tests_passed / cumulative_tests) if cumulative_tests else 0.0

    # Update progress bar postfix with current and running stats
    pbar.set_postfix({
      "task": res["task_id"],
      "status": res["status"],
      "t_s": f"{elapsed_seconds:.2f}",
      "avg_t_s": f"{running_avg_seconds:.2f}",
      "run_pass_rate": f"{running_pass_rate:.2%}",
      "solved_cur": int(solved_current),
      "solved_run": f"{cumulative_solved}/{idx+1}",
    })

    # Keep the existing per-sample print for detailed logs
    tqdm.write(
      f"{res['task_id']} {res['status']} {res['num_tests']} {res['num_tests_passed']} {elapsed_seconds:.2f}s solved:{cumulative_solved}/{idx+1}"
    )