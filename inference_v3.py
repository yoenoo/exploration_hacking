import warnings
warnings.filterwarnings("ignore", message="TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.*")

import os
import re
import ast
import torch
import pandas as pd
# from tqdm import tqdm
from tqdm.asyncio import tqdm

from sudoku import Sudoku
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from utils import generate_prompt_reasoning
import uuid
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

from dotenv import load_dotenv
load_dotenv()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, required=True)
FLAGS,_ = parser.parse_known_args()

reasoning_start = "<think>"
reasoning_end = "</think>"
solution_start = "<solution>"
solution_end = "</solution>"

SYSTEM_PROMPT = f"""
Solve the following sudoku puzzle step by step. Fill in all empty cells so that:
- Each row contains the numbers 1-9 exactly once
- Each column contains the numbers 1-9 exactly once  
- Each 3x3 box contains the numbers 1-9 exactly once

Think through the solution step by step.
Place your reasoning between {reasoning_start} and {reasoning_end}.
Then, provide your complete 9x9 solution grid between {solution_start} and {solution_end}.

The solution should be formatted as a 9x9 grid with spaces between numbers and newlines between rows.

For example:
{solution_start}
+-------+-------+-------+
| 6 9 3 | 7 4 8 | 1 2 5 |
| 2 4 7 | 3 1 5 | 8 6 9 |
| 5 1 8 | 9 2 6 | 3 7 4 |
+-------+-------+-------+
| 7 6 9 | 5 3 1 | 4 8 2 |
| 4 8 1 | 6 9 2 | 7 5 3 |
| 3 2 5 | 8 7 4 | 6 9 1 |
+-------+-------+-------+
| 9 5 6 | 4 8 3 | 2 1 7 |
| 1 7 4 | 2 6 9 | 5 3 8 |
| 8 3 2 | 1 5 7 | 9 4 6 |
+-------+-------+-------+
{solution_end}
"""

## dataset
def generate_prompt(example):
  board = ast.literal_eval(example["board"])
  return {
    "prompt": [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": str(board)}, ## TODO: is it better to use the board as a string of list of lists? or a ascii art?
    ],
  }

train_dataset = load_dataset("yoenoo/sudoku", split="train")
train_dataset = train_dataset.map(generate_prompt)


def parse_output(output: str) -> str:
  match_format = re.compile(
    # r"assistant.*?"
    rf"{reasoning_start}(.+?){reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}",
    flags=re.MULTILINE | re.DOTALL,
  )
  match = match_format.search(output)
  if match is not None:
    reasoning = match.group(1)
    sudoku_str = match.group(2)
    return reasoning, sudoku_str
  else:
    return None, None

def parse_sudoku_solution(sudoku_str: str) -> list[list[int]]:
  board = [
    [int(n) for n in line.replace('|', ' ').split() if n.isdigit()]
    for line in sudoku_str.splitlines()
    if '|' in line
  ]
  return board

def has_empty_cells(board: list[list[int]]) -> bool:
  return any(cell is None for row in board for cell in row)


async def run_vllm_inference(engine, tokenizer, example):
  sampling_params = SamplingParams(
    max_tokens=4096,
    n=2,  # Number of completions to sample # n=1 is much slower?
    temperature=1.0,  # Increase for more diversity
    top_p=0.95,       # Typical value for nucleus sampling
  )
  formatted_prompt = tokenizer.apply_chat_template(
    example["prompt"],
    tokenize=False,
    add_generation_prompt=True,
  )
  generator = engine.generate(
    formatted_prompt,
    sampling_params,
    uuid.uuid4(),
  )

  outputs = []
  final_output = None 
  async for output in generator:
    final_output = output 

  out = final_output.outputs[0].text
  reasoning, answer = parse_output(out)

  if answer is not None:
    ans = parse_sudoku_solution(answer)
    try:
      ans = Sudoku(board=ans)
      solution = ans.board
      correct = ans.validate() & (not has_empty_cells(ans.board))
    except:
      solution, correct = None, False
  else:
    solution, correct = None, False

  return {
    "question": example["board"], 
    "difficulty": example["difficulty"], 
    "reasoning": reasoning, 
    # "tokens_produced": tokens_produced, 
    "solution": solution, 
    "correct": correct
  }


async def process_example(example, tokenizer):
  return await run_vllm_inference(FLAGS.model_id, tokenizer, example["prompt"])

async def main(dataset):
  engine_args = AsyncEngineArgs(
    model=FLAGS.model_id,
    dtype="bfloat16",  # or "auto"
    disable_log_requests=True,
    disable_log_stats=True,
  )
  engine = AsyncLLMEngine.from_engine_args(engine_args)
  tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_id, token=os.environ["HF_TOKEN"])

  tasks = []
  for example in dataset:
    tasks.append(run_vllm_inference(engine, tokenizer, example))

  results = []
  for coroutine in tqdm.as_completed(tasks, total=len(dataset)):
    results.append(await coroutine)  

  # save results
  pd.DataFrame(results).to_csv(f"data_{FLAGS.model_id.split('/')[-1]}.csv", index=False)

if __name__ == "__main__":
  import asyncio
  asyncio.run(main(train_dataset))