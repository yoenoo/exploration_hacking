import sys; sys.path.append("sudoku")
import os
import re
import ast
import wandb
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from utils import generate_prompt_reasoning
from sudoku import Sudoku
from utils import is_valid_sudoku
from dotenv import load_dotenv
load_dotenv()

wandb.init(project="sudoku-trl")


## dataset
def generate_prompt(example):
  board = ast.literal_eval(example["board"])
  prompt = generate_prompt_reasoning(Sudoku(board=board))
  return {
    "prompt": [
      # {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": prompt},
    ],
  }

train_dataset = load_dataset("yoenoo/sudoku", split="train")
train_dataset = train_dataset.map(generate_prompt)


## model
model_id = "google/gemma-3-4b-it"
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  torch_dtype="auto",
  device_map="auto",
  token=os.environ["HF_TOKEN"],
)

lora_config = LoraConfig(
  task_type="CAUSAL_LM",
  r=8,
  lora_alpha=32,
  # lora_dropout=0.1,
  target_modules=["q_proj", "v_proj", "o_proj"],

  # r=16,
  # lora_alpha=32,
  # # target_modules="all-linear",
  # target_modules=["q_proj",
  #        "down_proj",
  #        "gate_proj",
  #        "o_proj",
  #        "k_proj",
  #        "v_proj",
  #        "up_proj"
  #        ]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


training_args = GRPOConfig(
  output_dir="GRPO",
  learning_rate=2e-5,
  remove_unused_columns=False, # to access the solution column in accuracy_reward
  gradient_accumulation_steps=2,
  num_train_epochs=1,
  bf16=True,
  # bf16=False,

  # Parameters that control de data preprocessing
  max_completion_length=512, ## 4096
  num_generations=8, 
  max_prompt_length=512, # default: 512

  # Parameters related to reporting and saving
  report_to=["wandb"],
  logging_steps=1,
  save_strategy="steps",
  save_steps=1,
)


def _match_solution(completion):
  matches = re.findall(r"<solution>(.*?)</solution>", completion, re.DOTALL | re.IGNORECASE)
  if not matches: return None
  match = ast.literal_eval(matches[-1])
  return match

def accuracy_reward(completions, **kwargs):
  # boards = kwargs["board"]
  diffs = kwargs["difficulty"]

  completion_contents = [c[0]["content"] for c in completions]
  matches = [_match_solution(c) for c in completion_contents]
  rewards = [is_valid_sudoku(m) if m is not None else None for m in matches] 
  print(rewards)
  updated_rewards = [r * d if r is not None else -0.1 for r, d in zip(rewards, diffs)] # small penalty for invalid format
  print(updated_rewards)
  return updated_rewards

trainer = GRPOTrainer(
  model=model,
  reward_funcs=[accuracy_reward], ## TODO: add format_reward?
  args=training_args,
  train_dataset=train_dataset,
)

trainer.train()