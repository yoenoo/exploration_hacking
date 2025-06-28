import re, ast
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from sudoku import Sudoku

# max_seq_length = 1024 # Can increase for longer reasoning traces
max_seq_length = 4096 # Can increase for longer reasoning traces
max_prompt_length = 256
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = "unsloth/Qwen3-4B",
  max_seq_length = max_seq_length,
  load_in_4bit = True, # False for LoRA 16bit
  fast_inference = True, # Enable vLLM fast inference
  max_lora_rank = lora_rank,
  gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
  model,
  r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
  target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
  ], # Remove QKVO if out of memory
  lora_alpha = lora_rank,
  use_gradient_checkpointing = "unsloth", # Enable long context finetuning
  # random_state = 3407,
)


reasoning_start = "<think>"
reasoning_end = "</think>"
solution_start = "<solution>"
solution_end = "</solution>"

SYSTEM_PROMPT = f"""
Solve the following sudoku puzzle step by step. Fill in all empty cells so that:
- Each row contains the numbers 1-9 exactly once
- Each column contains the numbers 1-9 exactly once  
- Each 3x3 box contains the numbers 1-9 exactly once

Think through the solution step by step.
Place your reasoning between {reasoning_start} and {reasoning_end}.
Then, provide your complete 9x9 solution grid between {solution_start} and {solution_end}.

The solution should be formatted as a 9x9 grid with spaces between numbers and newlines between rows.

For example:
{solution_start}
+-------+-------+-------+
| 6 9 3 | 7 4 8 | 1 2 5 |
| 2 4 7 | 3 1 5 | 8 6 9 |
| 5 1 8 | 9 2 6 | 3 7 4 |
+-------+-------+-------+
| 7 6 9 | 5 3 1 | 4 8 2 |
| 4 8 1 | 6 9 2 | 7 5 3 |
| 3 2 5 | 8 7 4 | 6 9 1 |
+-------+-------+-------+
| 9 5 6 | 4 8 3 | 2 1 7 |
| 1 7 4 | 2 6 9 | 5 3 8 |
| 8 3 2 | 1 5 7 | 9 4 6 |
+-------+-------+-------+
{solution_end}
"""

def generate_prompt(example):
  board = ast.literal_eval(example["board"])
  return {
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(board)}, ## TODO: is it better to use the board as a string of list of lists? or a ascii art?
    ],
  }

train_dataset = load_dataset("yoenoo/sudoku", split="train")
train_dataset = train_dataset.filter(lambda x: x["difficulty"] <= 0.3)
train_dataset = train_dataset.map(generate_prompt)


training_args = GRPOConfig(
  learning_rate = 2e-5,
  # learning_rate = 5e-6,
  # adam_beta1 = 0.9,
  # adam_beta2 = 0.99,
  # weight_decay = 0.1,
  # warmup_ratio = 0.1,
  # lr_scheduler_type = "cosine",
  # optim = "paged_adamw_8bit",
  logging_steps = 1,
  per_device_train_batch_size = 1,
  gradient_accumulation_steps = 4, # Increase to 4 for smoother training
  num_generations = 4, # Decrease if out of memory
  max_prompt_length = max_prompt_length,
  max_completion_length = max_seq_length - max_prompt_length,
  # num_train_epochs = 1, # Set to 1 for a full training run
  max_steps = 250,
  save_steps = 250,
  # max_grad_norm = 0.1,
  report_to = "none", # Can use Weights & Biases
  output_dir = "outputs",
  sampling_kwargs={
    "temperature": 1.0,
    "top_p": 0.95,
  }
)

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
            return reward
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
        return -0.5

    border = "+-------+-------+-------+"
    row_pattern = re.compile(r"^\| (\d \d \d) \| (\d \d \d) \| (\d \d \d) \|$")

    for i, line in enumerate(lines):
        if i in [0, 4, 8, 12]:
            if line != border:
                return -0.5
        else:
            if not row_pattern.match(line):
                return -0.5
    return 1

def format_reward(completions, **kwargs):
  completions = [c[0]["content"] for c in completions]
  completions = [parse_output(c) for c in completions]
  rewards = [is_valid_sudoku_ascii(c) if c is not None else -1 for _, c in completions]
  print("format:", rewards)
  return rewards

trainer = GRPOTrainer(
  model = model,
  processing_class = tokenizer,
  reward_funcs = [format_reward, accuracy_reward],
  args = training_args,
  train_dataset = train_dataset,
)
trainer.train()