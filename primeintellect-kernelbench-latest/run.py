from openai import OpenAI
from kernelbench import load_environment

from sandbox.runpod.orchestrator import KernelBenchOrchestrator
orchestrator = KernelBenchOrchestrator(
  gpu="NVIDIA GeForce RTX 3090",
  workers_max=3,
  max_poll_time=300,
  poll_interval=2,
  http_timeout=30.0,
  verbose=True,
)

env = load_environment(
  levels=[1],                # list or int
  max_samples=None,          # optional cap on dataset size
  subset_task_ids=["1"], #, "5", "10", "100"],      # e.g., ["1", "5", "10"]
  num_correctness_tests=5,   # trials for correctness
  speedup_threshold_fast1=1.0,
  speedup_threshold_fast2=2.0,
  random_seed=42,
  num_perf_trials=10,        # used for runtime timing and baseline
  reward_metric="gmsr_correct",
  orchestrator=orchestrator,
  # parallelize_eval=True,
)



sampling_args = {
  "temperature": 1.0,
  # "top_p": 0.95,
  "max_tokens": 10000,
}

results = env.evaluate(client=OpenAI(), model="gpt-5", rollouts_per_example=3, sampling_args=sampling_args)
# print(results)

# env.make_dataset(results).to_json("results.jsonl")

# # summary table
# print(results.info)
# print(results.metrics)


orchestrator._cleanup()