import os 
import sys
__DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(__DIR)

import time
import requests
import math
import json
import wandb
import shutil
import torch
import numpy as np 
from functools import lru_cache
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainerCallback

from src.train.dataset import build_dataset
from src.train.reward import KernelBenchReward
from src.train.utils import get_checkpoint_dir
from src.train.prompt import read_prompt
from src.wandb_utils.utils import wandb_init
from src.kernelbench_eval.utils import set_gpu_arch

from src.ox_rust.dataset import create_dataset
from src.ox_rust.prompt import SYSTEM_PROMPT
from src.ox_rust.rewards import RustCodeEvaluator, RewardFunctions


from dotenv import load_dotenv
load_dotenv()


import random
import numpy as np 
from vllm import SamplingParams

seed = 42
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

set_seed(seed)


def start_training_run(cfg):
  set_gpu_arch(cfg.gpu_arch) ## TODO: auto detect gpu arch

  # if cfg.clean_start:
  #   print("Starting fresh training run...")
  #   shutil.rmtree(cfg.io.original_src_dir, ignore_errors=True)
  #   shutil.rmtree(cfg.io.target_src_dir, ignore_errors=True)

  system_prompt = read_prompt(cfg.prompt.system_prompt) if cfg.prompt.system_prompt.startswith("src.train.prompts") else cfg.prompt.system_prompt

  run = wandb_init(cfg.project_name, cfg.run_name)
  dataset = create_dataset(cfg.dataset.path, SYSTEM_PROMPT)
  print(f"length of dataset: {len(dataset)}")

  peft_cfg = LoraConfig(
    r=cfg.lora.r, 
    lora_alpha=cfg.lora.lora_alpha, 
    lora_dropout=cfg.lora.lora_dropout, 
    bias=cfg.lora.bias,
    task_type=cfg.lora.task_type,
    target_modules=list(cfg.lora.target_modules),
  )

  training_args = GRPOConfig(
    output_dir=get_checkpoint_dir(cfg.project_name),

    temperature=cfg.grpo.temperature,
    top_p=cfg.grpo.top_p,
    learning_rate = cfg.grpo.lr,

    bf16=cfg.grpo.bf16,
    per_device_train_batch_size=cfg.grpo.per_device_train_batch_size,
    # generation_batch_size=cfg.grpo.generation_batch_size,
    gradient_accumulation_steps=cfg.grpo.gradient_accumulation_steps,

    use_vllm=cfg.grpo.vllm.use_vllm,
    vllm_mode=cfg.grpo.vllm.vllm_mode,
    vllm_server_base_url=cfg.grpo.vllm.vllm_server_base_url,
    vllm_gpu_memory_utilization=cfg.grpo.vllm.vllm_gpu_memory_utilization,
    
    num_generations=cfg.grpo.num_generations,
    max_prompt_length=cfg.grpo.max_prompt_length,
    max_completion_length=cfg.grpo.max_completion_length,
    # max_steps=cfg.grpo.max_steps,
    num_train_epochs=cfg.grpo.num_train_epochs,
    max_grad_norm=cfg.grpo.max_grad_norm,

    # multi-gpu stuff
    ddp_find_unused_parameters=cfg.grpo.ddp_find_unused_parameters,
    ddp_broadcast_buffers=cfg.grpo.ddp_broadcast_buffers,
    remove_unused_columns=cfg.grpo.remove_unused_columns,
    gradient_checkpointing=cfg.grpo.gradient_checkpointing,
    gradient_checkpointing_kwargs=dict(cfg.grpo.gradient_checkpointing_kwargs),

    # logging
    logging_strategy=cfg.grpo.logging_strategy,
    logging_steps=cfg.grpo.logging_steps,
    logging_first_step=cfg.grpo.logging_first_step,
    save_steps=cfg.grpo.save_steps,
    report_to=cfg.grpo.report_to,
    
    # do_eval=True,
    # eval_strategy="steps",
    # eval_steps=cfg.grpo.save_steps,
  )

  model = AutoModelForCausalLM.from_pretrained(
    cfg.model.name,
    dtype=cfg.model.torch_dtype,
    use_cache=cfg.model.use_cache,
  )
  tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
  if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


  evaluator = RustCodeEvaluator()
  rewards = RewardFunctions(evaluator)
  reward_functions = [
    rewards.create_reward_function("cargo_build", rewards.cargo_build_reward),
    rewards.create_reward_function("cargo_clippy", rewards.cargo_clippy_reward),
    rewards.create_reward_function("cargo_test", rewards.cargo_test_reward),
    rewards.create_reward_function("non_empty", rewards.non_empty_reward),
    rewards.create_reward_function("test_block", rewards.test_block_reward),
    rewards.create_reward_function("test_asserts", rewards.test_asserts_reward),
  ]

  trainer = GRPOTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_cfg,
    reward_funcs=reward_functions,
    # eval_dataset=eval_dataset,
    # callbacks=[
    #   EvalCallback(eval_dataset, tokenizer, eval_steps=1, max_completion_length=cfg.grpo.max_completion_length, wandb_run=run)
    # ],
  )

  ckpt = get_last_checkpoint(training_args.output_dir)
  if ckpt:
    with open(os.path.join(ckpt, "trainer_state.json")) as f:
      st = json.load(f)
    print("Resuming from:", ckpt, "global_step:", st.get("global_step"))
    trainer.train(resume_from_checkpoint=ckpt)
  else:
    print("No checkpoint found, starting fresh.")
    trainer.train()