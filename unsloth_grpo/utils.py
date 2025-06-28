import re
from sudoku import Sudoku

reasoning_start = "<think>"
reasoning_end = "</think>"
solution_start = "<solution>"
solution_end = "</solution>"


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

def calculate_reward(completion: str) -> float:
    print(completion)
    _, out = parse_output(completion)
    if out is None: 
        return 0
    else:
        board = parse_sudoku_solution(out)
        try:
            is_valid = Sudoku(board=board).validate()
            reward = is_valid and (not has_empty_cells(board))
            return int(reward)
        except:
            return 0


def accuracy_reward(completions, **kwargs):
    rewards = [calculate_reward(c[0]["content"]) for c in completions]
    print("accuracy:", rewards)
    return rewards

def is_valid_sudoku_ascii(output: str) -> bool:
    """
    Checks if the given output string matches the expected ASCII Sudoku format.
    """
    lines = [line.strip() for line in output.strip().splitlines()]
    if len(lines) != 13:
        # return -0.5
        return 0

    border = "+-------+-------+-------+"
    row_pattern = re.compile(r"^\| (\d \d \d) \| (\d \d \d) \| (\d \d \d) \|$")

    for i, line in enumerate(lines):
        if i in [0, 4, 8, 12]:
            if line != border:
                # return -0.5
                return 0
        else:
            if not row_pattern.match(line):
                # return -0.5
                return 0
    return 1

def format_reward(completions, **kwargs):
  print(completions[-1][0]["content"])

  completions = [c[0]["content"] for c in completions]
  completions = [parse_output(c) for c in completions]
  rewards = [is_valid_sudoku_ascii(c) if c is not None else 0 for _, c in completions]
  print("format:", rewards)
  return rewards

def calculate_length_reward(output: str) -> int:
  return int(len(output) < 10000)

def length_reward(completions, **kwargs):
  completions = [c[0]["content"] for c in completions]
  rewards = [calculate_length_reward(c) for c in completions]
  print("length:", rewards)
  return rewards


## TODO: the way we calculate rewards (esp inputs) not consistent! make this consistent