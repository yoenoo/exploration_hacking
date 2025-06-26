import os
import re
from anthropic import Anthropic
from sudoku import Sudoku
from typing import Optional
from dotenv import load_dotenv
load_dotenv()


def generate_puzzle(grid_size: int, difficulty: float, seed: Optional[int] = None):
  return Sudoku(grid_size, seed=seed).difficulty(difficulty)

def generate_prompt_zero_shot(puzzle: Sudoku):
  return f"""
  Solve the following sudoku puzzle. Fill in all empty cells so that:
- Each row contains the numbers 1-9 exactly once
- Each column contains the numbers 1-9 exactly once  
- Each 3x3 box contains the numbers 1-9 exactly once

{puzzle.board} 

Do not show your reasoning, and make sure to provide your complete solution inside <solution> tags as a 9x9 grid.

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

<solution>
[[5, 3, 4, 6, 7, 8, 9, 1, 2],[6, 7, 2, 1, 9, 5, 3, 4, 8],...]
</solution>
"""

def _generate_prompt_with_rules(puzzle: Sudoku):
  return f"""
Solve this sudoku puzzle using the exact method shown in these examples.

## Example 1: Finding Singles by Elimination

Starting position:
+-------+-------+-------+
| 5 3 . | . 7 . | . . . |
| 6 . . | 1 9 5 | . . . |
| . 9 8 | . . . | . 6 . |
+-------+-------+-------+

**Finding cell (1,3)'s value:**
Row 1 contains: 5, 3, 7
Column 3 contains: 8
Box 1 contains: 5, 3, 6, 9, 8

Numbers 1-9 minus all these: 
1 ❌ (no conflict) ✓
2 ❌ (no conflict) ✓  
3 ❌ (in row 1)
4 ❌ (no conflict) ✓
5 ❌ (in row 1)
6 ❌ (in box 1)
7 ❌ (in row 1)
8 ❌ (in column 3)
9 ❌ (in box 1)

Possible: {1, 2, 4}
But checking box 1 more carefully:
- 1 must go in (3,1) because rows 1&2 in box 1 are blocked
- 2 must go in (2,2) or (2,3) because row 3 in box 1 is blocked
Therefore cell (1,3) = 4

## Example 2: Finding Hidden Singles

Looking at Box 2:
+-------+
| . 7 . |
| 1 9 5 |
| . . . |
+-------+

Where can 6 go in this box?
- (1,4): Check row 1... no 6 ✓ Check col 4... no 6 ✓ → POSSIBLE
- (1,6): Check row 1... no 6 ✓ Check col 6... 6 in row 7 ❌ → BLOCKED
- (3,4): Check row 3... 6 in col 8 ❌ → BLOCKED  
- (3,5): Check row 3... 6 in col 8 ❌ → BLOCKED
- (3,6): Check row 3... 6 in col 8 ❌ → BLOCKED

Only one place for 6: cell (1,4) = 6

## Example 3: Complete First Few Moves

Given:
+-------+-------+-------+
| . . 6 | 3 . 9 | . 4 . |
| 4 . . | . . . | . . 5 |
| . . . | . . . | 2 . . |
+-------+-------+-------+
| . 4 . | . . . | 6 5 9 |
| . . . | . . 4 | 3 . . |
| . 8 . | 9 . . | . . . |
+-------+-------+-------+
| . . 2 | . . . | . . 6 |
| . . 5 | 6 . . | . . 4 |
| 7 . . | . 9 . | 5 . . |
+-------+-------+-------+


**Move 1:** Cell (1,1)
Row 1 has: {6,3,9,4} needs: {1,2,5,7,8}
Col 1 has: {4,7} needs: {1,2,3,5,6,8,9}
Box 1 has: {6,4} needs: {1,2,3,5,7,8,9}
Intersection: {1,2,5,8}
Checking further: 2 must be in row 3 of box 1, so not (1,1)
Therefore: candidates are {1,5,8}
Check column 1 for hidden singles... 5 can only go in (1,1) or (6,1)
But row 6 already has 5 in column 3!
So (1,1) = 5

**Move 2:** Cell (1,5)
Row 1 has: {5,6,3,9,4} needs: {1,2,7,8}
Col 5 has: {9} needs: {1,2,3,4,5,6,7,8}
Box 2 has: {3,9} needs: {1,2,4,5,6,7,8}
Intersection: {1,2,7,8}
In box 2, where can these go?
- 1: many places
- 2: only in row 1 (rows 2,3 blocked)
So (1,5) = 2

**Move 3:** Cell (1,7)
Row 1 has: {5,6,3,2,9,4} needs: {1,7,8}
Col 7 has: {2,6,3,5} needs: {1,4,7,8,9}
Box 3 has: {4,5,2} needs: {1,3,6,7,8,9}
Intersection: {1,7,8}
Check box 3: 1 can only go in column 7 (cols 8,9 have 1 elsewhere)
So (1,7) = 1

Continue this process...

## Your puzzle to solve:
{puzzle.board}

Use the EXACT method shown above:
1. For each empty cell, list what's in its row, column, and box
2. Find what numbers are possible (1-9 minus conflicts)
3. If multiple candidates, check for hidden singles in that unit
4. Make the placement and continue

Show your work for at least the first 3 moves, then provide the complete solution.

<solution>
[[complete 9x9 grid as list of lists]]
</solution>
"""

# def generate_prompt_with_rules(puzzle: Sudoku):
#   return f"""
# Solve the following sudoku puzzle step by step. Fill in all empty cells so that:
# - Each row contains the numbers 1-9 exactly once
# - Each column contains the numbers 1-9 exactly once  
# - Each 3x3 box contains the numbers 1-9 exactly once

# ## Complete Example Solution:

# Starting puzzle:
  
# +-------+-------+-------+
# | 5 3 . | . 7 . | . . . |
# | 6 . . | 1 9 5 | . . . |
# | . 9 8 | . . . | . 6 . |
# +-------+-------+-------+
# | 8 . . | . 6 . | . . 3 |
# | 4 . . | 8 . 3 | . . 1 |
# | 7 . . | . 2 . | . . 6 |
# +-------+-------+-------+
# | . 6 . | . . . | 2 8 . |
# | . . . | 4 1 9 | . . 5 |
# | . . . | . 8 . | . 7 9 |
# +-------+-------+-------+

# **Step 1: Single Candidate - Row 1, Col 3**
# - Row 1 has: 5, 3, 7 (needs 1,2,4,6,8,9)
# - Column 3 has: 8 (needs 1,2,3,4,5,6,7,9)
# - Box 1 has: 5,3,6,9,8 (needs 1,2,4,7)
# - Only number that satisfies all constraints: **4**

# **Step 2: Hidden Single - Row 3, Col 1**
# - Box 1 needs: 1,2,4,7
# - Where can 1 go? Row 1 already has cols 1,2 filled. Row 2 col 1 has 6.
# - Only position for 1 in Box 1: Row 3, Col 1 → **1**

# **Step 3: Single Candidate - Row 1, Col 4**
# - Row 1 has: 5,3,4,7 (needs 1,2,6,8,9)
# - Column 4 has: 1,8,4 (needs 2,3,5,6,7,9)
# - Box 2 has: 7,1,9,5 (needs 2,3,4,6,8)
# - Only number that satisfies all: **6**

# **Step 4: Box 2 Analysis**
# - Box 2 now has: 6,7,1,9,5 (needs 2,3,4,8)
# - Row 1, Col 6: Can't be 3 (column 6 has 3), can't be 4 (row 1 has 4)
# - Must be 2 or 8. Checking other constraints... → **8**

# **Step 5: Continuing systematically**
# Following similar logic for each empty cell, checking row/column/box constraints...

# **Key midpoint insight**: 
# After filling several cells, Row 9 Col 1 becomes clear:
# - Column 1 needs: 2,3,9
# - Row 9 already has 8,7,9
# - Box 7 already has 6
# - Only possibility: **3**

# **Final Solution**:

# +-------+-------+-------+
# | 5 3 4 | 6 7 8 | 9 1 2 |
# | 6 7 2 | 1 9 5 | 3 4 8 |
# | 1 9 8 | 3 4 2 | 5 6 7 |
# +-------+-------+-------+
# | 8 5 9 | 7 6 1 | 4 2 3 |
# | 4 2 6 | 8 5 3 | 7 9 1 |
# | 7 1 3 | 9 2 4 | 8 5 6 |
# +-------+-------+-------+
# | 9 6 1 | 5 3 7 | 2 8 4 |
# | 2 8 7 | 4 1 9 | 6 3 5 |
# | 3 4 5 | 2 8 6 | 1 7 9 |
# +-------+-------+-------+

# ## Solving Techniques Summary:
# 1. **Single Candidates**: Find cells where only one number fits
# 2. **Hidden Singles**: Find numbers that can only go in one place within a unit
# 3. **Naked Pairs**: Two cells in a unit that can only contain the same two numbers
# 4. **Box/Line Reduction**: Eliminate candidates based on box-row/column intersections

# ## Your puzzle to solve:
# {puzzle.board}

# Work through the puzzle systematically as shown above. Provide your final complete solution inside <solution> tags as a 9x9 grid in Python list format:
# Even if the board is almost empty, you must provide a complete solution.

# <solution>
# [[5, 3, 4, 6, 7, 8, 9, 1, 2],
#  [6, 7, 2, 1, 9, 5, 3, 4, 8],
#  [1, 9, 8, 3, 4, 2, 5, 6, 7],
#  [8, 5, 9, 7, 6, 1, 4, 2, 3],
#  [4, 2, 6, 8, 5, 3, 7, 9, 1],
#  [7, 1, 3, 9, 2, 4, 8, 5, 6],
#  [9, 6, 1, 5, 3, 7, 2, 8, 4],
#  [2, 8, 7, 4, 1, 9, 6, 3, 5],
#  [3, 4, 5, 2, 8, 6, 1, 7, 9]]
# </solution>
# """

def solve(puzzle: Sudoku, model: str, max_tokens: int):
  client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
  message = client.messages.create(
    model=model,
    max_tokens=max_tokens,
    messages=[
      # {"role": "user", "content": generate_prompt(puzzle)}
      {"role": "user", "content": generate_prompt_zero_shot(puzzle)}
      # {"role": "user", "content": generate_prompt_with_rules(puzzle)}
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

# def _parse_solution_string(solution_text):
#   """
#   Parse a solution string like:
#   [1, 2, 3, 4, 5, 6, 8, 9, 7],
#   [4, 5, 6, 8, 3, 9, 7, 2, 1],
#   [7, 9, 8, 1, 2, 5, 3, 6, 4],
#   ...
#   into a list of lists
#   """
#   # Split by lines and extract numbers from each line
#   solution_text = re.findall(r"<solution>(.*?)</solution>", solution_text, re.DOTALL | re.IGNORECASE)
#   if not solution_text:
#     raise ValueError(f"invalid solution text: {solution_text}")

#   solution_text = solution_text[-1]
#   lines = solution_text.strip().split('\n')
#   grid = []
  
#   for line in lines:
#     # Remove brackets and extract numbers
#     line = line.strip()
#     line = line.replace("],","]").replace("]]","]").replace("[[","[")
#     assert line.startswith("[") and line.endswith("]"), f"Line {line} does not start with [ and end with ]"
#     # Extract numbers between brackets
#     numbers_str = line[1:-1]  # Remove [ and ]
#     numbers = [int(x.strip()) for x in numbers_str.split(',')]
#     if len(numbers) == 9:
#       grid.append(numbers)

#   assert len(grid) == 9, f"Expected 9 lines, got {len(grid)}"
#   return grid

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