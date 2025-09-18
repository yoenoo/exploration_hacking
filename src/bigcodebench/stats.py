import json
import os
import re
import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

with open("eval_results.jsonl", "r") as f:
  results = [json.loads(line) for line in f]

stats = defaultdict(lambda: defaultdict(list))
overall_tasks = defaultdict(list)
for res in results:
  cats = res.get("category")
  if not isinstance(cats, list):
    cats = [cats]
  task_id = res["task_id"]
  passed = (res["status"] == "pass")
  overall_tasks[task_id].append(passed)
  for cat in cats:
    stats[cat][task_id].append(passed)

for category, task_map in stats.items():
  for task_id, passes in task_map.items():
    passes_sorted = sorted(passes)
    pass_k = np.cumsum(passes_sorted) / np.arange(1, len(passes_sorted) + 1)
    print(f"[{category}] {task_id}: {pass_k}")

# Aggregate and plot overall and per-category pass@k bar charts
def compute_pass_k_avg_over_tasks(task_map):
  # Average pass@k over tasks, including only tasks with at least k candidates
  if not task_map:
    return np.array([])
  per_task_passk = {}
  k_max = 0
  for task_id, passes in task_map.items():
    arr = np.array(sorted(passes), dtype=float)
    pass_k = np.cumsum(arr) / np.arange(1, len(arr) + 1)
    per_task_passk[task_id] = pass_k
    if len(pass_k) > k_max:
      k_max = len(pass_k)
  # For each k, average across tasks with length >= k
  avg = []
  for k in range(1, k_max + 1):
    vals = [pk[k - 1] for pk in per_task_passk.values() if len(pk) >= k]
    if vals:
      avg.append(float(np.mean(vals)))
    else:
      avg.append(0.0)
  return np.array(avg)

def plot_pass_k_bar(pass_k_array, title, out_path):
  if pass_k_array.size == 0:
    return
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  x = np.arange(1, len(pass_k_array) + 1)
  plt.figure(figsize=(10, 4))
  plt.bar(x, pass_k_array * 100.0, color="#4C78A8")
  plt.ylim(0.0, 100.0)
  plt.xlabel("k")
  plt.ylabel("Accuracy (%)")
  plt.title(title)
  plt.tight_layout()
  plt.savefig(out_path)
  plt.close()

# Overall (no double counting tasks)
overall_pass_k = compute_pass_k_avg_over_tasks(overall_tasks)
plot_pass_k_bar(overall_pass_k, "Overall pass@k", os.path.join("plots", "pass_k_overall.png"))

# Per category
for category, task_map in stats.items():
  cat_pass_k = compute_pass_k_avg_over_tasks(task_map)
  safe_cat = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(category))
  plot_pass_k_bar(cat_pass_k, f"pass@k [{category}]", os.path.join("plots", f"pass_k_{safe_cat}.png"))