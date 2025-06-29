import warnings
warnings.filterwarnings("ignore", message="TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.*")

import os
import re
import ast
import torch
import pandas as pd
from tqdm.asyncio import tqdm

from transformers import AutoTokenizer
from datasets import load_dataset

import uuid
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

from dotenv import load_dotenv
load_dotenv()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, required=True)
FLAGS,_ = parser.parse_known_args()


async def run_vllm_inference(engine, tokenizer, example, max_tokens, temperature=1.0, top_p=0.95):
  sampling_params = SamplingParams(
    max_tokens=max_tokens,
    n=2,                      # Number of completions to sample # n=1 is much slower?
    temperature=temperature,  # Increase for more diversity
    top_p=top_p,              # Typical value for nucleus sampling
  )

  # formatted_prompt = tokenizer.apply_chat_template(
  #   example["prompt"],
  #   tokenize=False,
  #   add_generation_prompt=True,
  # )
  formatted_prompt = example["prompt"]

  generator = engine.generate(formatted_prompt, sampling_params, uuid.uuid4())

  outputs = []
  final_output = None 
  async for output in generator:
    final_output = output 

  out = final_output.outputs[0].text
  return out

# async def process_example(example, tokenizer, **kwargs):
#   return await run_vllm_inference(FLAGS.model_id, tokenizer, example["prompt"], **kwargs)

async def main(dataset):
  engine_args = AsyncEngineArgs(
    model=FLAGS.model_id,
    dtype="bfloat16", 
    disable_log_requests=True,
    disable_log_stats=True,
    # gpu_memory_utilization=0.2,
  )
  engine = AsyncLLMEngine.from_engine_args(engine_args)
  tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_id, token=os.environ["HF_TOKEN"])

  tasks = []
  for example in dataset:
    tasks.append(run_vllm_inference(engine, tokenizer, example, max_tokens=4096, temperature=1.0, top_p=0.95))
    if len(tasks) > 3:
      break

  results = []
  for coroutine in tqdm.as_completed(tasks, total=len(dataset)):
    results.append(await coroutine)  

  print(results)
  # # save results
  # pd.DataFrame(results).to_csv(f"data_{FLAGS.model_id.split('/')[-1]}.csv", index=False)

if __name__ == "__main__":
  from datasets import load_dataset
  train_dataset = load_dataset("predibase/wordle-grpo", split="train")

  import asyncio
  asyncio.run(main(train_dataset))