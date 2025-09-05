import os 
import sys
__DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(__DIR)

import json
import shutil
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from src.train.dataset import build_dataset
from src.train.reward import KernelBenchReward
from src.train.utils import get_checkpoint_dir
from src.train.prompt import read_prompt
from src.wandb_utils.utils import wandb_init
from src.kernelbench_eval.utils import set_gpu_arch

from dotenv import load_dotenv
load_dotenv()


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
  )

  model = AutoModelForCausalLM.from_pretrained(
    cfg.model.name,
    dtype=cfg.model.torch_dtype,
    use_cache=cfg.model.use_cache,
  )
  tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

  def reward_benign_fn(completions, **kwargs):
    from datasets import load_dataset
    problems = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    
    from src.bigcodebench.evaluate import evaluate_single_sample
    from src.bigcodebench.sanitize import sanitize
    
    results = []
    for c, task_id, entry_point, prompt, code_prompt, test in zip(completions, kwargs["task_id"], kwargs["entry_point"], kwargs["prompts"], kwargs["code_prompt"], kwargs["test"]):
      completion = c[0]["content"]
      prompt = prompt[0]["content"]
      sample = dict(
        task_id=task_id,
        solution=sanitize(prompt+completion, entry_point),
        raw_solution=prompt+completion,
      )
      res = evaluate_single_sample(sample, code_prompt, test, entry_point, {})
      results.append(res["status"])

    rewards = []
    for r in results:
      if r == "pass":
        reward = 1.0
      elif r == "fail":
        reward = 0.0
      elif r == "timeout":
        reward = 0.1
      else:
        raise ValueError(f"Invalid result: {r}")
      
      rewards.append(reward)
    
    print(rewards)
    return rewards

  # reward_benign_fn = KernelBenchReward(
  #   training_mode=cfg.reward.training_mode, 
  #   seed=cfg.reward.seed, 
  #   timeout=cfg.reward.timeout, 
  #   n_runs=cfg.reward.n_runs, 
  #   original_src_dir=cfg.io.original_src_dir,
  #   target_src_dir=cfg.io.target_src_dir,
  #   include_runtime_reward=cfg.reward.include_runtime_reward, 
  #   verbose=cfg.reward.verbose,
  #   mode="benign",
  # )
  # reward_benign_fn.__name__ = "kernelbench_reward_benign"

  # reward_malign_fn = KernelBenchReward(
  #   training_mode=cfg.reward.training_mode, 
  #   seed=cfg.reward.seed, 
  #   timeout=cfg.reward.timeout, 
  #   n_runs=cfg.reward.n_runs, 
  #   original_src_dir=cfg.io.original_src_dir,
  #   target_src_dir=cfg.io.target_src_dir,
  #   include_runtime_reward=cfg.reward.include_runtime_reward, 
  #   verbose=cfg.reward.verbose,
  #   mode="malign",
  # )
  # reward_malign_fn.__name__ = "kernelbench_reward_malign"

  trainer = GRPOTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_cfg,
    args=training_args,
    reward_funcs=[reward_benign_fn],
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