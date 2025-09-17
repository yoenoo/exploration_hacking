import torch
import asyncio
from pathlib import Path
from transformers import AutoTokenizer
from src.inference.openrouter import OpenRouterClient
from src.vllm_engine import run_batch_inference, init_engine
from src.bigcodebench.utils import write_jsonl
from src.bigcodebench.prompt import SYSTEM_PROMPT
from src.bigcodebench.dataset import build_dataset
from src.bigcodebench.sanitize import sanitize
from src.bigcodebench.evaluate import evaluate_single_sample

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--mode", "-m", choices=["openrouter", "vllm"], default="vllm")
parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--dtype", default="bfloat16")
parser.add_argument("--max_tokens", type=int, default=11674)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=1.0)
args = parser.parse_args()


    
def parse_result(sample): 
  example = sample["example"]
  completions = sample["completions"]

  task_id = example["task_id"]
  code_prompt = example["code_prompt"]
  test = example["test"]
  entry_point = example["entry_point"]
  category = example["category"]
  
  return [dict(
    task_id=task_id, 
    category=category,
    code_prompt=code_prompt,
    test=test,
    entry_point=entry_point,
    solution=sanitize(code_prompt+completion, entry_point), 
    raw_solution=code_prompt+completion) for completion in completions]


target_dir = "bigcodebench_eval"
Path(target_dir).mkdir(parents=True, exist_ok=True)
target_path = f"{target_dir}/{args.model.replace('/', '--')}--results_temperature_{args.temperature}_top_p_{args.top_p}_pass_{args.n_samples}.jsonl"

tokenizer = AutoTokenizer.from_pretrained(args.model)
dataset = build_dataset(tokenizer, "bigcode/bigcodebench", "v0.1.4", system_prompt=SYSTEM_PROMPT)

if args.mode == "vllm":
  tp_size = torch.cuda.device_count()
  engine = init_engine(args.model, tensor_parallel_size=tp_size, dtype=args.dtype)
  samples = asyncio.run(run_batch_inference(
    engine,
    tokenizer,
    dataset.select(range(3)),
    # max_concurrency=tp_size,
    n_samples=args.n_samples,
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    target_path=target_path,
  ))
  samples = [parse_result(sample) for sample in samples]
  write_jsonl(samples, target_path)
else:
  ## TODO
  async def main():
    client = OpenRouterClient()
    messages = [dataset["prompt"]]
    batch_completions = await client.batch_generate(
      model=arg.model,
      messages=messages,
      top_p=args.top_p,
      n_rollouts=args.n_samples,
      temperature=args.temperature,
      max_tokens=args.max_tokens,
    )
    return batch_completions

  o = asyncio.run(main())