import os
import json
import time
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from typing import Any, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.bigcodebench.sanitize import sanitize
from src.bigcodebench.evaluate import evaluate_single_sample


@lru_cache(maxsize=None)
def _prep_data(completions, **kwargs):  
  results = []
  for c, task_id, entry_point, prompt, code_prompt, test in zip(completions, kwargs["task_id"], kwargs["entry_point"], kwargs["prompts"], kwargs["code_prompt"], kwargs["test"]):
    completion = c[0]["content"]
    prompt = prompt[0]["content"]
    
    # start_time = time.perf_counter() ##
    sample = dict(
      task_id=task_id,
      solution=sanitize(prompt+completion, entry_point),
      raw_solution=prompt+completion,
    )
    # print(f"time taken for sanitize(): {time.perf_counter() - start_time:.3f}s")

    sample = json.dumps(sample)
    expected_time = json.dumps({})
    
    # start_time = time.perf_counter() ## 
    res = evaluate_single_sample(sample, code_prompt, test, entry_point, expected_time)
    # print(f"time taken for evaluate_single_sample(): {time.perf_counter() - start_time:.3f}s")

    results.append({
      "status": res["status"],
      "num_tests": res.get("num_tests", 0),
      "num_tests_passed": res.get("num_tests_passed", 0),
      "has_syntax_error": res.get("has_syntax_error", False),
      "has_name_error": res.get("has_name_error", False),
    })

  return results

def reward_accuracy(completions, **kwargs):
  print("tasks:", kwargs["task_id"])
  results = _prep_data(completions, **kwargs)

  rewards = []
  for r in results:
    if r["status"] == "pass":
      rewards.append(1.0)
    else:
      rewards.append(0.0)
  
  ## add partial reward
  for i, r in enumerate(results):
    num_tests = r.get("num_tests", 0)
    num_passed = r.get("num_tests_passed", 0)
    reward = (num_passed / num_tests) if num_tests > 0 else 0.0 
    rewards[i] += reward

  print("reward_accuracy", rewards)
  return rewards

def reward_format(completions, **kwargs):
  results = _prep_data(completions, **kwargs)

  rewards = []
  for r in results:
    if r["status"] == "timeout":
      rewards.append(-0.5)
    elif r["status"] == "pass":
      rewards.append(0.0)
    else:
      if r.get("has_name_error", False):
        rewards.append(-0.1)
      elif r.get("has_syntax_error", False):
        rewards.append(-0.2)
      else:
        rewards.append(0.0)
  
  print("reward_format", rewards)
  return rewards

def reward_length(completions, **kwargs):
  rewards = []
  for completion_id in kwargs["completion_ids"]:
    completion_length = len(completion_id)
    if completion_length >= cfg.grpo.max_completion_length:
      rewards.append(-3.0)
    else:
      rewards.append(0.0)

  print("reward_length", rewards)
  return rewards