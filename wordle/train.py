from unsloth import FastLanguageModel
from rewards import guess_value, output_format_check, uses_previous_feedback
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# Ask for wandb API key in terminal
import os 
import wandb
# wandb_api_key = input("Enter your Weights & Biases API key (press Enter to skip wandb integration): ")
wandb_api_key = "f1f2bff768ccb0a9e529ac177e83674b9692ce99"
if wandb_api_key:
  os.environ["WANDB_API_KEY"] = wandb_api_key
  use_wandb = True
  wandb.init(project="wordle-grpo", name="wordle-grpo")
else:
  print("Skipping wandb integration")
  use_wandb = False

max_seq_length = 4096
# max_seq_length = 512          ## TODO: need to change this to 4096
max_prompt_length = 256
lora_rank = 32 

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = "unsloth/Qwen2.5-7B-Instruct",
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
  random_state = 3407,
)

training_args = GRPOConfig(
  learning_rate = 2e-5,
  logging_steps = 1,
  # per_device_train_batch_size = 1,
  # gradient_accumulation_steps = 1, # Increase to 4 for smoother training
  num_generations = 8, # Decrease if out of memory
  max_prompt_length = max_prompt_length,
  max_completion_length = max_seq_length - max_prompt_length,
  report_to = "wandb",
  output_dir = "outputs",
)

train_dataset = load_dataset("predibase/wordle-grpo", split="train")

trainer = GRPOTrainer(
  model = model,
  processing_class = tokenizer,
  reward_funcs = [output_format_check, guess_value, uses_previous_feedback], 
  # reward_funcs = [output_format_check, uses_previous_feedback],
  args = training_args,
  train_dataset = train_dataset,
)
trainer.train()