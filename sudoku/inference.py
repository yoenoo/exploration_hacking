import os
import re
import ast
import torch
import pandas as pd
from tqdm import tqdm
from sudoku import Sudoku
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from utils import generate_prompt_reasoning

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore", message="TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.*")


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
train_dataset = train_dataset.filter(lambda x: x["difficulty"] <= 0.03)
train_dataset = train_dataset.map(generate_prompt)

def parse_output(output: str) -> str:
  match_format = re.compile(
    r"assistant.*?"
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

## model
# model_id = "google/gemma-3-4b-it"
model_id = "Qwen/Qwen3-4B"
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  torch_dtype="auto",
  device_map="auto",
  token=os.environ["HF_TOKEN"],
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

data = []

counter = 0
for item in tqdm(train_dataset):
  inputs = tokenizer.apply_chat_template(
    item["prompt"],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
  ).to(model.device).to(torch.bfloat16)

  with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=4096)
    input_tokens = inputs.input_ids.shape[-1]
    output_tokens = outputs.shape[-1]
    tokens_produced = output_tokens - input_tokens
    print("Tokens Produced:", tokens_produced)

    outputs = tokenizer.batch_decode(outputs)
    reasoning, sudoku_str = parse_output(outputs[0])

    if sudoku_str is not None:
      ans = parse_sudoku_solution(sudoku_str)
      ans = Sudoku(board=ans)
      correct = ans.validate() & (not has_empty_cells(ans.board))
    else:
      correct = False

    data.append({"question": item["board"], "difficulty": item["difficulty"], "reasoning": reasoning, "tokens_produced": tokens_produced, "solution": ans.board, "correct": correct})
    # print(data[-1])

  counter += 1
  if counter > 3:
    break

# save the data
df = pd.DataFrame(data)
df.to_csv(f"data_{model_id.split('/')[-1]}.csv", index=False)