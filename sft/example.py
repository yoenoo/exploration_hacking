from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import torch

dataset = load_dataset("predibase/wordle-sft", split="train")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.bfloat16)

training_args = SFTConfig(output_dir="/tmp")

peft_config = LoraConfig(
  r=16,
  lora_alpha=32,
  lora_dropout=0.05,
  target_modules="all-linear",
  # modules_to_save=["lm_head", "embed_token"],
  task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
  model,
  train_dataset=dataset,
  args=training_args,
  peft_config=peft_config,
)

trainer.train()