import os
from collections import defaultdict
from utils import generate_puzzle, solve, correctly_filled_cells

# TODO : generate rollouts
n_tries = 10
grid_size = 3
# difficulties = [0.02, 0.1, 0.2, 0.3, 0.5, 0.9]
difficulties = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
difficulties = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# 0.02 -> only 1 cell to fill
# 0.99 -> only 1 cell is filled




from sudoku import Sudoku
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", token=os.getenv("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", token=os.getenv("HF_TOKEN"))


results = defaultdict(list)
for difficulty in difficulties:
  print(f"Sudoku Difficulty: {difficulty}")
  # for _ in range(n_tries):
  while len(results[difficulty]) < n_tries:
    puzzle = generate_puzzle(grid_size, difficulty)
    # answer, correct = solve(puzzle, model="claude-sonnet-4-20250514", max_tokens=4096)
    # answer, correct = solve(puzzle, model="claude-3-5-haiku-20241022", max_tokens=1024)
    try:
      answer, correct = solve(puzzle, model, tokenizer, max_tokens=4096)
      results[difficulty].append({"puzzle": puzzle, "answer": answer, "correct": correct})
      print(results[difficulty])
    except SyntaxError:
      continue

  acc = sum(r["correct"] for r in results[difficulty]) / n_tries
  print(f"Accuracy: {acc:.4f}")

  perc_correct = sum([correctly_filled_cells(r["answer"], r["puzzle"].solve().board, r["puzzle"].board) for r in results[difficulty]]) / n_tries
  print(f"% Correct: {perc_correct:.4f}")