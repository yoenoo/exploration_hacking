import json
import numpy as np
import pandas as pd 
from tqdm import tqdm
from sudoku import Sudoku


# make a huggingface dataset
# question only dataset

def _check_has_empty_cells(board: list[list[int]]) -> bool:
  for row in board:
    for cell in row:
      if cell is None:
        return True
  return False

def _check_has_nonempty_cells(board: list[list[int]]) -> bool:
  for row in board:
    for cell in row:
      if isinstance(cell, int) and cell is not None:
        return True
  return False

# def format_board(board: list[list[int]]) -> str:
#   for i in range(len(board)):
#     for j in range(len(board[i])):
#       board[i][j] = board[i][j] if board[i][j] is not None else "x"
#   return board

def generate_puzzle(difficulty: float, size: int = 10):
  puzzles = []
  while len(puzzles) < size:
    puzzle = Sudoku(3, seed=None).difficulty(difficulty)
    if _check_has_empty_cells(puzzle.board) and _check_has_nonempty_cells(puzzle.board) and puzzle.board not in puzzles:
      puzzles.append(puzzle.board)
  
  puzzles = [{"board": p, "difficulty": difficulty} for p in puzzles]
  return puzzles


if __name__ == "__main__":
  # dataset
  n_samples = 100
  difficulties = [x / 100 for x in range(2,100)] + [0.999, 0.9999]
  # difficulties = list(np.arange(0.02, 1, 0.01)) + [0.999, 0.9999]
  
  ds = []
  for d in tqdm(difficulties):
    ds.extend(generate_puzzle(d, n_samples))

  # ideally, none of the boards are the same
  # assert len(set(p[0] for p in ds)) == len(difficulties) * n_samples

  pd.DataFrame(ds).to_csv(f"sudoku-10k.csv", index=False)

  # with open("sudoku.jsonl", "w") as f:
  #   for d in ds:
  #     f.write(json.dumps(d) + "\n")