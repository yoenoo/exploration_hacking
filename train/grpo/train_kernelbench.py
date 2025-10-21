import verifiers as vf

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

trainer = vf.GRPOTrainer(
  mode=model,
  processing_class=tokenizer,
  env=env,
  args=args,
)
trainer.train()
