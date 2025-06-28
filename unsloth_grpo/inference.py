from unsloth import FastLanguageModel
from vllm import SamplingParams
from datasets import load_dataset
import ast
import torch

max_seq_length = 4096
lora_rank = 32
model_name = "unsloth/Qwen3-4B"
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = model_name,
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
train_dataset = train_dataset.map(generate_prompt)

if model_name == "unsloth/Qwen3-4B":
  text = tokenizer.apply_chat_template(
    train_dataset[100]["prompt"], ## harder task
    tokenize = False, 
    add_generation_prompt = True
  )
  print(text)
  sampling_params = SamplingParams(
    temperature = 1.0,
    top_p = 0.95,
    max_tokens = max_seq_length,
  )
  output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
  )[0].outputs[0].text
  print(output)

  from utils import is_valid_sudoku_ascii, calculate_reward
  print(is_valid_sudoku_ascii(output))
  print(calculate_reward(output))

  # ================================

elif model_name == "unsloth/gemma-3-4b-it":
  from unsloth.chat_templates import get_chat_template
  tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
  )
  text = tokenizer.apply_chat_template(
    train_dataset[0]["prompt"],
    add_generation_prompt = True, # Must add for generation
  )
  from transformers import TextStreamer
  _ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = max_seq_length, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
  )
else:
  raise ValueError(f"Model {model} not supported")


if torch.distributed.is_initialized():
  torch.distributed.destroy_process_group()