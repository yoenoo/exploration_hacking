import json
from collections import defaultdict

stats = defaultdict(list)
with open("results.jsonl") as f:
  for line in f.readlines():
    line = json.loads(line)
    info = line["info"]
    task_name = info["name"]
    stats[task_name].append(line["metric_gmsr_correct"])
    #stats[task_name][1].append(line["metric_fast_0"])
    #stats[task_name][2].apend(line["metric_fast_1"])
    #stats[task_name][3].append(line["metric_fast_2"])
    #stats[task_name][4].append(line["metric_speedup"])

print(stats)
