import os
import torch
from utils import generate_puzzle, generate_prompt_reasoning
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()


puzzle = generate_puzzle(grid_size=3, difficulty=0.02)
prompt = generate_prompt_reasoning(puzzle)
# print(puzzle)
# print(prompt)


from sudoku import Sudoku
o = Sudoku(board=[[6, 8, 9, 1, 2, 3, 7, 4, 5],
 [2, 3, 1, 7, 5, 4, 9, 6, 8],
 [4, 5, 7, 8, 9, 6, 1, 2, 3],
 [8, 2, 5, 3, 6, 7, 4, 9, 1],
 [3, 7, 4, 9, 1, 5, 6, 8, 2],
 [1, 9, 6, 2, 4, 8, 3, 5, 7],
 [7, 6, 3, 5, 8, 9, 2, 1, 4],
 [9, 1, 8, 4, 3, 2, 5, 7, 6],
 [5, 4, 2, 6, 7, 1, 8, 3, 9]]).validate()
print(o)

# so Gemma3-4b can solve 3x3 sudoku..!
# can you make the inference fast enough? on consumer GPUs?

exit()

HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", token=HF_TOKEN)

messages = [
  [
    # {
    #   "role": "system",
    #   "content": [{"type": "text", "text": "You are a sudoku solver."},]
    # },
    {
      "role": "user",
      "content": [{"type": "text", "text": prompt},]
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
  outputs = model.generate(**inputs, max_new_tokens=4096)

outputs = tokenizer.batch_decode(outputs)
print(outputs[0])