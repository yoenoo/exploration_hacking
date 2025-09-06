from typing import Any, Dict, Optional
from src.bigcodebench.utils import untrusted_check


def evaluate_single_sample(
  sample: Dict[str, Any],
  code_prompt: str,
  # problems: Dict[str, Dict[str, Any]],
  test: str,
  entry_point: str,
  expected_time: Dict[str, Optional[float]],
  # calibrated: bool = True,
  # max_as_limit: int = 30 * 1024,
  max_as_limit: int = 512 * 1024,
  # max_data_limit: int = 30 * 1024,
  max_data_limit: int = 512 * 1024,
  # max_stack_limit: int = 10,
  max_stack_limit: int = 512,
  min_time_limit: float = 1.0,
  default_gt_time_limit: float = 20.0,
  include_solution: bool = True,
) -> Dict[str, Any]:
  task_id = sample["task_id"]
  # problem = problems[task_id]
  # problem = problems.filter(lambda x: x["task_id"] == task_id).to_list()[0] ## this raises an OOM error

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
    # test_code=problem["test"],
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
    "details": details,
  }
  if include_solution:
    record["solution"] = solution

  return record