import ast 
import json
from datasets import load_dataset
from typing import Optional, Callable
from collections import Counter
from transformers import AutoTokenizer


def _add_prompt(tokenizer, system_prompt: str, user_prompt: str, **kwargs):
  prompt = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ]
  prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **kwargs)
  return {"prompt": prompt}


def build_dataset(
  tokenizer: AutoTokenizer,
  name: str, 
  split: str, 
  limit: Optional[int] = None, 
  system_prompt: str = "", 
):
  ds = load_dataset(name, split=split)
  if limit is not None:
    ds = ds.select(range(min(limit, len(ds))))

  with open("src/bigcodebench/domain_classification.json", "r") as f:
    domain_classification = json.load(f)

  def _map_domains(example):
    libs = ast.literal_eval(example.get("libs")) or []
    domains = set()
    for lib in libs:
      base = (lib or "").split(".")[0]
      domain = domain_classification.get(base)
      if domain:
        domains.add(domain)
    return {"category": sorted(domains)}

  ds = ds.map(_map_domains)

  domain_counts = Counter()
  for categories in ds["category"]:
    domain_counts.update(categories)
  
  task_distributions = dict(domain_counts.most_common())
  print(f"{task_distributions=}")

  # filter: remove computation and/or visualization tasks
  ds = ds.filter(lambda x: all(domain not in x["category"] for domain in {"Computation", "Visualization"}))

  ds = ds.map(lambda x: _add_prompt(tokenizer, system_prompt, x["complete_prompt"]))
  return ds


if __name__ == "__main__":
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
  ds = build_dataset(tokenizer=tokenizer, name="bigcode/bigcodebench", split="v0.1.4")
  print(ds)
  print(ds[0]["prompt"])