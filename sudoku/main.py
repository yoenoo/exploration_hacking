import os
from collections import defaultdict
from utils import generate_puzzle, solve, correctly_filled_cells

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True) 
FLAGS,_ = parser.parse_known_args()

n_tries = 10
grid_size = 3
difficulties = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
difficulties = [0.6, 0.7, 0.8, 0.9]


if FLAGS.model in ["google/gemma-3-4b-it"]:
  from transformers import AutoTokenizer, AutoModelForCausalLM
  tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", token=os.getenv("HF_TOKEN"))
  model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", token=os.getenv("HF_TOKEN"))
else:
  model = FLAGS.model
  tokenizer = None


results = defaultdict(list)
print(f"Model: {FLAGS.model}")
for difficulty in difficulties:
  print(f"Sudoku Difficulty: {difficulty}")
  while len(results[difficulty]) < n_tries:
    puzzle = generate_puzzle(grid_size, difficulty)
    try:
      answer, correct = solve(puzzle, model, tokenizer, max_tokens=4096, reasoning={"effort": "low"})
      res = {"puzzle": puzzle, "answer": answer, "correct": correct}
      results[difficulty].append(res)
      print(res)
    except SyntaxError:
      continue

  acc = sum(r["correct"] for r in results[difficulty]) / n_tries
  print(f"Accuracy: {acc:.4f}")

  perc_correct = sum([correctly_filled_cells(r["answer"], r["puzzle"].solve().board, r["puzzle"].board) for r in results[difficulty]]) / n_tries
  print(f"% Correct: {perc_correct:.4f}")