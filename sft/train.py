from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dataset = load_dataset("predibase/wordle-sft", split="train")

peft_config = LoraConfig(
  r=4,
  lora_alpha=8,
  lora_dropout=0.05,
  # target_modules="all-linear",
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
  task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
  output_dir="Qwen3-4B-SFT",
  report_to="wandb",
)

model_id = "Qwen/Qwen3-4B"
model_kwargs = dict(
  torch_dtype=torch.bfloat16,
  device_map="auto",
  trust_remote_code=True,
  # **kwargs,
)
model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)

trainer = SFTTrainer(
  model=model,
  args=sft_config,
  peft_config=peft_config,
  train_dataset=dataset,
)

trainer.train()



exit()

import torch
from trl import SFTTrainer
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.utils.quantization_config import BitsAndBytesConfig


# _TRAINING_ARGS = dict(
#   per_device_train_batch_size=1,  # TODO: make batched training work
#   # per_device_eval_batch_size=1,
#   # gradient_accumulation_steps=2,
#   optim="paged_adamw_32bit",
#   num_train_epochs=5,
#   logging_steps=0.05,
#   # warmup_steps=5,
#   # logging_strategy="steps",
#   learning_rate=1e-3,
#   # fp16=False,
#   bf16=True,
#   # group_by_length=True,
#   report_to="wandb",
# )

_TRAINING_ARGS = dict(
  report_to="wandb",
)

_LORA_CONFIG = LoraConfig(
  r=16,  # TODO: change?
  lora_alpha=32,
  lora_dropout=0.05,
  bias="none",
  target_modules="all-linear",  # TODO: change?
  task_type="CAUSAL_LM",
)


def _load_model(model_id: str, quantize: bool = False, **kwargs):
  model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    **kwargs,
  )

  if quantize:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_quant_storage=torch.bfloat16,
    )

  model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
  tokenizer = AutoTokenizer.from_pretrained(model_id)

  model.config.use_cache = False
  model.config.pretraining_tp = 1

  return model, tokenizer


def train_model(
  model_id: str,
  dataset_path: str,
  adapter_output_dir: str,
  merged_output_dir: str,
  training_kwargs: dict = {},
  model_kwargs: dict = {},
):
  model, tokenizer = _load_model(model_id, **model_kwargs)
  model = get_peft_model(model, _LORA_CONFIG)

  # data_collator = DataCollatorForLanguageModeling(
  #   tokenizer=tokenizer,
  #   mlm=False,
  # )
  train_dataset = load_dataset(dataset_path)
  print(train_dataset)

  args = TrainingArguments(
    output_dir=adapter_output_dir,
    **(_TRAINING_ARGS | training_kwargs),
  )

  trainer_kwargs = dict(
    model=model,
    args=args,
    train_dataset=train_dataset,
    # data_collator=data_collator,
    peft_config=model.peft_config["default"],
  )

  trainer = SFTTrainer(**trainer_kwargs)
  trainer.train()
  trainer.save_model()

  # Free the memory
  del model
  del trainer
  torch.cuda.empty_cache()

  # Merge the adapter into the model
  merged = AutoPeftModelForCausalLM.from_pretrained(
    adapter_output_dir
  ).merge_and_unload()
  merged.save_pretrained(merged_output_dir)
  tokenizer.save_pretrained(merged_output_dir)

  # Free the memory again
  del merged
  del tokenizer
  torch.cuda.empty_cache()


if __name__ == "__main__":
  model_id = "Qwen/Qwen3-4B"
  dataset_path = "predibase/wordle-sft"
  train_model(model_id, dataset_path=dataset_path, adapter_output_dir="./adapter", merged_output_dir="./merged")