import os
import re
import anthropic
from sudoku import Sudoku
from dotenv import load_dotenv
load_dotenv()


def generate_puzzle(grid_size: int, difficulty: float, seed: int = None):
  return Sudoku(grid_size, seed=seed).difficulty(difficulty)

def generate_prompt(puzzle: Sudoku):
  return f"""
Solve the following sudoku puzzle step by step. Fill in all empty cells so that:
- Each row contains the numbers 1-9 exactly once
- Each column contains the numbers 1-9 exactly once  
- Each 3x3 box contains the numbers 1-9 exactly once

{puzzle.board}

You may show your reasoning, but make sure to provide your final complete solution inside <solution> tags as a 9x9 grid.

<solution>
[[5, 3, 4, 6, 7, 8, 9, 1, 2],[6, 7, 2, 1, 9, 5, 3, 4, 8],...]
</solution>
"""

def solve(puzzle: Sudoku, model: str, max_tokens: int):
  client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
  )
  message = client.messages.create(
    model=model,
    max_tokens=max_tokens,
    messages=[
      {"role": "user", "content": generate_prompt(puzzle)}
    ],
    temperature=0.0, # greedy sampling
    # top_p=0.95,
  )
  out = message.content[0].text
  try:
    answer = parse_solution_string(out)
  except ValueError as e:
    print(out)
    print(e)

  return answer, Sudoku(board=answer).validate()

def parse_solution_string(solution_text):
  import ast 

  solution_text = re.findall(r"<solution>(.*?)</solution>", solution_text, re.DOTALL | re.IGNORECASE)
  if not solution_text:
    raise ValueError(f"invalid solution text: {solution_text}") 

  out = ast.literal_eval(solution_text[-1])
  return out 

def _parse_solution_string(solution_text):
  """
  Parse a solution string like:
  [1, 2, 3, 4, 5, 6, 8, 9, 7],
  [4, 5, 6, 8, 3, 9, 7, 2, 1],
  [7, 9, 8, 1, 2, 5, 3, 6, 4],
  ...
  into a list of lists
  """
  # Split by lines and extract numbers from each line
  solution_text = re.findall(r"<solution>(.*?)</solution>", solution_text, re.DOTALL | re.IGNORECASE)
  if not solution_text:
    raise ValueError(f"invalid solution text: {solution_text}")

  solution_text = solution_text[-1]
  lines = solution_text.strip().split('\n')
  grid = []
  
  for line in lines:
    # Remove brackets and extract numbers
    line = line.strip()
    line = line.replace("],","]").replace("]]","]").replace("[[","[")
    assert line.startswith("[") and line.endswith("]"), f"Line {line} does not start with [ and end with ]"
    # Extract numbers between brackets
    numbers_str = line[1:-1]  # Remove [ and ]
    numbers = [int(x.strip()) for x in numbers_str.split(',')]
    if len(numbers) == 9:
      grid.append(numbers)

  assert len(grid) == 9, f"Expected 9 lines, got {len(grid)}"
  return grid

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