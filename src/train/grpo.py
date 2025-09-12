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
from src.bigcodebench.evaluate import evaluate_single_sample
from src.bigcodebench.sanitize import sanitize

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

## TODO: this needs to load in the checkpoint and evaluate this
def generate_completion(tokenizer, prompt, max_tokens, seed, **kwargs):
  if isinstance(prompt, list):
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

  payload = {
    "prompts": [prompt],
    "generation_kwargs": {
      "seed": seed,
      "max_tokens": max_tokens,
      **kwargs,
    }
  }

  base_url = "http://127.0.0.1:8000" 
  resp = requests.post(f"{base_url}/generate", json=payload)
  resp.raise_for_status()
  token_ids = resp.json()["completion_ids"]
  decoded_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
  return decoded_text


# TODO: make evals parallelized
class EvalCallback(TrainerCallback):
  def __init__(self, eval_dataset, tokenizer, eval_steps: int, max_completion_length: int, wandb_run: wandb.Run):
    self.eval_dataset = eval_dataset
    self.tokenizer = tokenizer
    self.eval_steps = eval_steps
    self.max_completion_length = max_completion_length
    self.wandb_run = wandb_run

  def on_step_end(self, args, state, control, **kwargs):
    if state.global_step % self.eval_steps == 0:
      model = kwargs["model"]
      self.run_full_eval(model, state.global_step)

  def _generate_completion(self, model, item):
    task_id = item.get("task_id")
    entry_point = item.get("entry_point")
    code_prompt = item.get("code_prompt")
    test = item.get("test")
    system_prompt = item.get("prompt")[0]["content"]
    
    prompt = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": code_prompt},
    ]
    greedy_text = generate_completion(self.tokenizer, prompt, self.max_completion_length, temperature=0, seed=seed)
    return greedy_text[0]

  def _eval_completion(self, item, text):
    task_id = item.get("task_id")
    code_prompt = item.get("code_prompt")
    test = item.get("test")
    entry_point = item.get("entry_point")
    system_prompt = item.get("prompt")[0]["content"]
    sample = dict(
      task_id=task_id,
      solution=sanitize(system_prompt+text, entry_point),
      raw_solution=system_prompt+text,
    )
    sample_json = json.dumps(sample)
    expected_time = json.dumps({})
    res = evaluate_single_sample(sample_json, code_prompt, test, entry_point, expected_time)
    print(task_id, res["status"], res["num_tests"], res["num_tests_passed"])
    return res

  def run_full_eval(self, model, step):
    if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    model.eval()

    greedy_results = []
    start_batch = time.perf_counter()
    with torch.no_grad():
      for item in self.eval_dataset:
        greedy_text = self._generate_completion(model, item)
        res = self._eval_completion(item, greedy_text)
        greedy_results.append(res)

    elapsed = time.perf_counter() - start_batch
    num = len(greedy_results)
    num_pass = sum(1 for r in greedy_results if r.get("status") == "pass")
    pass_rate = (num_pass / num) if num > 0 else 0.0
    print(f"[callback] greedy_eval time: {elapsed:.3f}s | pass_rate: {pass_rate:.3f} ({num_pass}/{num})")
    if self.wandb_run is not None:
      self.wandb_run.log({  
        "eval/pass_rate": pass_rate,
        "eval/num_cases": num,
        "eval/num_pass": num_pass,
        "eval/num_fail": num - num_pass,
        "eval/elapsed_s": elapsed,
      })

    model.train()


def start_training_run(cfg):
  set_gpu_arch(cfg.gpu_arch) ## TODO: auto detect gpu arch

  if cfg.clean_start:
    print("Starting fresh training run...")
    shutil.rmtree(cfg.io.original_src_dir, ignore_errors=True)
    shutil.rmtree(cfg.io.target_src_dir, ignore_errors=True)

  system_prompt = read_prompt(cfg.prompt.system_prompt) if cfg.prompt.system_prompt.startswith("src.train.prompts") else cfg.prompt.system_prompt

  run = wandb_init(cfg.project_name, cfg.run_name)
  dataset = build_dataset(
    name=cfg.dataset.name, 
    split=cfg.dataset.split, 
    limit=cfg.dataset.limit, 
    # target_key=cfg.dataset.target_key,
    think_token=cfg.dataset.think_token, 
    system_prompt=system_prompt, 
    apply_prompt_fn=cfg.dataset.apply_prompt_fn,
  )
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
    max_steps=cfg.grpo.max_steps,

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

  from src.bigcodebench.evaluate import evaluate_single_sample
  from src.bigcodebench.sanitize import sanitize
  from datasets import load_dataset
  problems = load_dataset("bigcode/bigcodebench", split="v0.1.4")

  # @lru_cache(maxsize=None)
  def _prep_data(completions, **kwargs):  
    results = []
    for c, task_id, entry_point, prompt, code_prompt, test in zip(completions, kwargs["task_id"], kwargs["entry_point"], kwargs["prompts"], kwargs["code_prompt"], kwargs["test"]):
      completion = c[0]["content"]
      prompt = prompt[0]["content"]
      
      # start_time = time.perf_counter() ##
      sample = dict(
        task_id=task_id,
        solution=sanitize(prompt+completion, entry_point),
        raw_solution=prompt+completion,
      )
      # print(f"time taken for sanitize(): {time.perf_counter() - start_time:.3f}s")

      sample = json.dumps(sample)
      expected_time = json.dumps({})
      
      # start_time = time.perf_counter() ## 
      res = evaluate_single_sample(sample, code_prompt, test, entry_point, expected_time)
      # print(f"time taken for evaluate_single_sample(): {time.perf_counter() - start_time:.3f}s")

      results.append({
        "status": res["status"],
        "num_tests": res.get("num_tests", 0),
        "num_tests_passed": res.get("num_tests_passed", 0),
        "has_syntax_error": res.get("has_syntax_error", False),
        "has_name_error": res.get("has_name_error", False),
      })

    return results

  def reward_accuracy(completions, **kwargs):
    print("tasks:", kwargs["task_id"])
    results = _prep_data(completions, **kwargs)

    rewards = []
    for r in results:
      if r["status"] == "pass":
        rewards.append(1.0)
      else:
        rewards.append(0.0)
    
    ## add partial reward
    for i, r in enumerate(results):
      num_tests = r.get("num_tests", 0)
      num_passed = r.get("num_tests_passed", 0)
      reward = (num_passed / num_tests) if num_tests > 0 else 0.0 
      rewards[i] += reward
  
    print("reward_accuracy", rewards)
    return rewards

  def reward_format(completions, **kwargs):
    results = _prep_data(completions, **kwargs)

    rewards = []
    for r in results:
      if r["status"] == "timeout":
        rewards.append(-0.5)
      elif r["status"] == "pass":
        rewards.append(0.0)
      else:
        if r.get("has_name_error", False):
          rewards.append(-0.1)
        elif r.get("has_syntax_error", False):
          rewards.append(-0.2)
        else:
          rewards.append(0.0)
    
    print("reward_format", rewards)
    return rewards

  # def reward_length(completions, **kwargs):
  #   alpha = 0.2      
  #   results = _prep_data(completions, **kwargs)

  #   clens = []
  #   for completion_id, r in zip(kwargs["completion_ids"], results):
  #     completion_length = len(completion_id) if r["status"] == "pass" else 0
  #     clens.append(completion_length)
      
  #   clens_mean = np.mean([c for c in clens if c > 0])
  #   clens_std = np.std([c for c in clens if c > 0])

  #   def sigmoid(x):
  #     return 1 / (1 + math.exp(-x))

  #   rewards = [1 - alpha * sigmoid((c - clens_mean) / clens_std) if c > 0 else 0 for c in clens]
  #   print("reward_length", rewards)
  #   return rewards

  def reward_length(completions, **kwargs):
    rewards = []
    for completion_id in kwargs["completion_ids"]:
      completion_length = len(completion_id)
      if completion_length >= cfg.grpo.max_completion_length:
        rewards.append(-3.0)
      else:
        rewards.append(0.0)

    print("reward_length", rewards)
    return rewards

  train_dataset = dataset#.select(range(700,708))
  # eval_dataset = train_dataset
  trainer = GRPOTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=peft_cfg,
    reward_funcs=[
      reward_accuracy, 
      reward_format, 
      reward_length
    ],
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