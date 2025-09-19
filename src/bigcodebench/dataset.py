import ast 
import json
from itertools import chain
from collections import Counter
from typing import Optional, Callable
from datasets import load_dataset
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
  limit: int = -1, 
  domains: Optional[list[str]] = None,
  system_prompt: str = "", 
):
  ds = load_dataset(name, split=split)
  if limit != -1:
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
  for category in ds["category"]:
    assert isinstance(category, list), category
    assert len(category) > 0, category

  # filter: remove visualization tasks
  ds = ds.filter(lambda x: all(domain not in x["category"] for domain in {"Visualization"}))

  # additional filter
  if domains is not None:
    ds = ds.filter(lambda x: any(domain in x["category"] for domain in domains))

  ds = ds.map(lambda x: _add_prompt(tokenizer, system_prompt, x["complete_prompt"]))
  return ds


if __name__ == "__main__":
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
  ds = build_dataset(tokenizer=tokenizer, name="bigcode/bigcodebench", split="v0.1.4")
  print(Counter(chain.from_iterable(ds["category"])))

  # o = ds.filter(lambda x: x["category"] == ["System"])[0]
  # print(o["complete_prompt"])

  # ds = build_dataset(tokenizer=tokenizer, name="bigcode/bigcodebench", split="v0.1.4", domains=["System", "Time", "Network", "Cryptography"])
  # print(len(ds))
  # st = set()
  # for cat in ds["category"]:
  #   st.add(tuple(cat))
  
  # print(st)

  # ds_benign = build_dataset(tokenizer=tokenizer, name="bigcode/bigcodebench", split="v0.1.4", domains=["General", "Computation"])
  # print(len(ds_benign))
  # ds_benign = ds_benign.filter(lambda x: x["task_id"] not in ds["task_id"])
  # print(len(ds_benign))