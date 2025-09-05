import asyncio
from pathlib import Path
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

dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")
dataset = dataset.map(lambda x: {"prompt": apply_chat_template(x["complete_prompt"])})


target_dir = "bcb_results"
Path(target_dir).mkdir(parents=True, exist_ok=True)
target_path = f"{target_dir}/{model.replace('/', '--')}--bigcodebench_eval_results.jsonl"

engine = init_engine(model, tensor_parallel_size=1, dtype="bfloat16")

samples = asyncio.run(run_batch_inference(
  engine,
  tokenizer,
  dataset,
  n_samples=3,
  max_tokens=16384,
  target_path=target_path,
  parse_fn=lambda e, cs: print(cs[0]),
))