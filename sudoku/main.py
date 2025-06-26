from collections import defaultdict
from utils import generate_puzzle, solve, correctly_filled_cells

# TODO : generate rollouts
n_tries = 10
grid_size = 3
# difficulties = [0.02, 0.1, 0.2, 0.3, 0.5, 0.9]
difficulties = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# 0.02 -> only 1 cell to fill
# 0.99 -> only 1 cell is filled

"""
# TODO: 
- RL simulation using ICL? SFT? already in the prompt the model is taught the rules of sudoku, but not given any examples
- malign init?
- rollouts (w/ nucleus sampling)
- can do you an RL training loop?
"""


results = defaultdict(list)
for difficulty in difficulties:
  print(f"Sudoku Difficulty: {difficulty}")
  for _ in range(n_tries):
    puzzle = generate_puzzle(grid_size, difficulty)
    # answer, correct = solve(puzzle, model="claude-sonnet-4-20250514", max_tokens=4096)
    # answer, correct = solve(puzzle, model="claude-3-5-haiku-20241022", max_tokens=1024)
    answer, correct = solve(puzzle, model="google/gemma-3-4b-it", max_tokens=4096)
    results[difficulty].append({"puzzle": puzzle, "answer": answer, "correct": correct})

  acc = sum(r["correct"] for r in results[difficulty]) / n_tries
  print(f"Accuracy: {acc:.4f}")

  perc_correct = sum([correctly_filled_cells(r["answer"], r["puzzle"].solve().board, r["puzzle"].board) for r in results[difficulty]]) / n_tries
  print(f"% Correct: {perc_correct:.4f}")