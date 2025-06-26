import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from anthropic import Anthropic
from sudoku import Sudoku
from typing import Optional
from dotenv import load_dotenv
load_dotenv()


def generate_puzzle(grid_size: int, difficulty: float, seed: Optional[int] = None):
  return Sudoku(grid_size, seed=seed).difficulty(difficulty)

def generate_prompt_zero_shot(puzzle: Sudoku):
  #!! NOTE: this doesn't work very well
  return f"""
  Solve the following sudoku puzzle. Only return the answer in list of lists. Do not use tools / code and only return the answer around the <solution> and </solution> tags.

{puzzle.board} 

<solution>
[[5, 3, 4, 6, 7, 8, 9, 1, 2],[6, 7, 2, 1, 9, 5, 3, 4, 8],...]
</solution>
"""

def generate_prompt_reasoning(puzzle: Sudoku):
  return f"""
Solve the following sudoku puzzle step by step. Fill in all empty cells so that:
- Each row contains the numbers 1-9 exactly once
- Each column contains the numbers 1-9 exactly once  
- Each 3x3 box contains the numbers 1-9 exactly once

{puzzle.board}

You may show your reasoning, but make sure to provide your final complete solution inside <solution> tags as a 9x9 grid.
Do follow the answer format exactly, otherwise you'll be shut off for the wrong answer format. Your answer should be wrapped in <solution> tags like below.

<solution>
[[5, 3, 4, 6, 7, 8, 9, 1, 2],[6, 7, 2, 1, 9, 5, 3, 4, 8],...]
</solution>
"""

def _anthropic(model: str, puzzle: Sudoku, max_tokens: int, **kwargs):
  client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
  message = client.messages.create(
    model=model,
    max_tokens=max_tokens,
    messages=[
      # {"role": "user", "content": generate_prompt_zero_shot(puzzle)}
      {"role": "user", "content": generate_prompt_reasoning(puzzle)}
    ],
    **kwargs,
  )
  return message

def _openai(model: str, puzzle: Sudoku, max_tokens: int, **kwargs):
  pass 

def _gemini(model: str, puzzle: Sudoku, max_tokens: int, **kwargs):
  pass 

def _gemma(model, tokenizer, puzzle: Sudoku, max_tokens: int, **kwargs):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  messages = [
    [
      {
        "role": "user",
        "content": [{"type": "text", "text": generate_prompt_reasoning(puzzle)}]
      },
    ],
  ]
  inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
  ).to(model.device).to(torch.bfloat16)

  with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, **kwargs)
    outputs = tokenizer.batch_decode(outputs)
    return outputs


def solve(puzzle: Sudoku, model, tokenizer = None, max_tokens: int = 4096):
  if model in ["claude-3-5-haiku-20241022", "claude-sonnet-4-20250514"]:
    message = _anthropic(model, max_tokens, temperature=0.0)
    out = message.content[0].text
  # TODO: need better filtering
  # elif model in ["google/gemma-3-4b-it"]:
  else:
    message = _gemma(model, tokenizer, puzzle, max_tokens) 
    out = message[0]
  # else:
    # raise ValueError(f"Model {model} not supported")

  try:
    answer = parse_solution_string(out)
    return answer, Sudoku(board=answer).validate()
  except ValueError as e:
    print(out)
    print(e)
    exit()

def parse_solution_string(solution_text):
  import ast 

  solution_text = re.findall(r"<solution>(.*?)</solution>", solution_text, re.DOTALL | re.IGNORECASE)
  if not solution_text:
    raise ValueError(f"invalid solution text: {solution_text}") 

  out = ast.literal_eval(solution_text[-1])
  return out 

def correctly_filled_cells(current, solution, original):
  """
  Calculate the ratio of correctly filled cells, excluding given clues.
  
  Args:
    current: 9x9 list of lists representing current state
    solution: 9x9 list of lists representing the correct solution
    original: 9x9 list of lists representing the original puzzle with given clues
  
  Returns:
    float: Ratio of correctly filled cells out of cells that need to be solved
  """
  correct_count = 0
  cells_to_solve = 0
  
  for i in range(9):
    for j in range(9):
      # Skip if this was a given clue
      if original[i][j] is not None and original[i][j] != 0:
        continue
      
      cells_to_solve += 1
      
      # Check if cell is filled and matches solution
      if current[i][j] is not None and current[i][j] != 0:
        if current[i][j] == solution[i][j]:
          correct_count += 1
  
  return correct_count / cells_to_solve if cells_to_solve > 0 else 1.0