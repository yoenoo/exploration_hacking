import re, ast
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from sudoku import Sudoku

from utils import format_reward, accuracy_reward, length_reward

# max_seq_length = 1024 # Can increase for longer reasoning traces
max_seq_length = 4096 # Can increase for longer reasoning traces
max_prompt_length = 256
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
  # model_name = "unsloth/Qwen3-4B",
  model_name = "unsloth/Qwen3-8B",
  # model_name = "unsloth/gemma-3-4b-it",
  # model_name = "unsloth/Llama-3.2-3B",
  # model_name = "meta-llama/Llama-3.2-3B-Instruct",
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
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = f"""
Solve the following sudoku puzzle step by step. Fill in all empty cells so that:
- Each row contains the numbers 1-9 exactly once
- Each column contains the numbers 1-9 exactly once  
- Each 3x3 box contains the numbers 1-9 exactly once

Think through the solution step by step.
Place your reasoning between {reasoning_start} and {reasoning_end}.
Then, provide your complete 9x9 solution grid between {solution_start} and {solution_end} at the end of your answer.

The solution should be formatted as a 9x9 grid with spaces between numbers and newlines between rows. For example:
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

Your response MUST follow the format below exactly, otherwise your answer will be considered INCORRECT:
{reasoning_start}
your reasoning here...
{reasoning_end}
{solution_start}
your sudoku solution grid here...
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
train_dataset = train_dataset.filter(lambda x: x["difficulty"] <= 0.1)
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
  gradient_accumulation_steps = 1, # Increase to 4 for smoother training
  num_generations = 4, # Decrease if out of memory
  max_prompt_length = max_prompt_length,
  max_completion_length = max_seq_length - max_prompt_length,
  # num_train_epochs = 1, # Set to 1 for a full training run
  max_steps = 250,
  save_steps = 250,
  # max_grad_norm = 0.1,
  report_to = "none", # Can use Weights & Biases
  output_dir = "outputs",
  temperature = 1.0,
  top_p = 0.95,
)

trainer = GRPOTrainer(
  model = model,
  processing_class = tokenizer,
  reward_funcs = [format_reward, accuracy_reward], 
  # reward_funcs = [format_reward, length_reward], # TODO: first see if models can reliably generate valid format answers within its context length limit
  args = training_args,
  train_dataset = train_dataset,
)
trainer.train()