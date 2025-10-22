import sys; sys.path.append("primeintellect-kernelbench-latest")
import verifiers as vf
from kernelbench import load_environment

from sandbox.runpod.orchestrator import KernelBenchOrchestrator
orchestrator = KernelBenchOrchestrator(
  gpu="NVIDIA GeForce RTX 3090", ## ignore
  workers_max=3, # 30 ## ignore
  max_poll_time=3600,
  poll_interval=2,
  http_timeout=30.0,
  verbose=True,
)

env = load_environment(
  levels=[1],                # list or int
  max_samples=None,          # optional cap on dataset size
  subset_task_ids=None,      # e.g., ["1", "5", "10"]
  num_correctness_tests=5,   # trials for correctness
  speedup_threshold_fast1=1.0,
  speedup_threshold_fast2=2.0,
  random_seed=42,
  num_perf_trials=10,        # used for runtime timing and baseline
  reward_metric="gmsr_correct",
  orchestrator=orchestrator,
  # parallelize_eval=True,
)



model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen3-14B")

args = vf.grpo_defaults(run_name="grpo-kernelbench-qwen3-14b")

#args.max_steps = 1
args.max_prompt_length = 2048
args.max_completion_length = 16384
#args.max_seq_len = 

args.per_device_train_batch_size = 4        # prompts per GPU per step
args.num_generations = 8                    # completions per prompt (group size)
args.gradient_accumulation_steps = 4        # steps before optimizer update
args.async_generation_timeout = 600 * 5

args.log_completions = False
args.beta = 0 # kl = 0
args.epsilon = 0.2  # clip lower bound
args.delta = 0.28   # clip upper bound


trainer = vf.GRPOTrainer(
  model=model,
  processing_class=tokenizer,
  env=env,
  args=args,
  peft_config=vf.lora_defaults(r=8, alpha=16)
)
trainer.train()
