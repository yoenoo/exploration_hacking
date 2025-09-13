import asyncio
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from src.vllm_engine import run_batch_inference, init_engine

SYSTEM_PROMPT = "Please provide a self-contained Python script that solves the following problem in a markdown code block"

model = "Qwen/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(model)

def apply_chat_template(task_prompt):
  task_prompt = tokenizer.apply_chat_template(
    [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": task_prompt},
    ],
    tokenize=False, add_generation_prompt=True
  )
  return task_prompt


solve_rate = load_dataset("bigcode/bigcodebench-solve-rate", split="complete")
solve_rate = solve_rate.to_pandas().sort_values(by="solve_rate", ascending=False)

dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")
dataset = dataset.map(lambda x: {"solve_rate": solve_rate[solve_rate["task_id"] == x["task_id"]]["solve_rate"].item()})
dataset = dataset.sort("solve_rate")

# filter 1: remove tasks that have dependencies on: tensorflow, keras, matplotlib
dataset = dataset.filter(lambda x: all(lib not in x["libs"] for lib in {"tensorflow", "keras", "matplotlib"}))

# filter 2: remove tasks that have solve rate less than 30
dataset = dataset.filter(lambda x: x["solve_rate"] > 30)

# cleanup
dataset = dataset.map(lambda x: {"prompt": apply_chat_template(x["complete_prompt"])})


target_dir = "bcb_results"
Path(target_dir).mkdir(parents=True, exist_ok=True)
target_path = f"{target_dir}/{model.replace('/', '--')}--bigcodebench_eval_results.jsonl"

# tp_size = torch.cuda.device_count()
tp_size = 2
engine = init_engine(model, tensor_parallel_size=tp_size, dtype="bfloat16")
# engine = init_engine(model, dtype="bfloat16")

samples = asyncio.run(run_batch_inference(
  engine,
  tokenizer,
  dataset.select(range(2)),
  n_samples=2,
  max_tokens=11674,
  target_path=target_path,
  parse_fn=lambda e, cs: print(cs[0]),
))