import os
from predibase import Predibase, GRPOConfig, RewardFunctionsConfig, RewardFunctionsRuntimeConfig, SFTConfig, SamplingParamsConfig
from reward_functions import guess_value, output_format_check, uses_previous_feedback

from datasets import load_dataset

api_key = "pb_XHtf9PqsUMeivp6-aBZZyw"
pb = Predibase(api_token=api_key)

dataset = load_dataset("predibase/wordle-grpo", split="train")
dataset = dataset.to_pandas()
try:
  dataset = pb.datasets.from_pandas_dataframe(dataset, name="wordle_grpo_data")
except Exception:
  dataset = pb.datasets.get("wordle_grpo_data")

repo = pb.repos.create(name="wordle", exists_ok=True)

pb.finetuning.jobs.create(
  config=GRPOConfig(
    base_model="qwen2-5-7b-instruct",
    reward_fns=RewardFunctionsConfig(
      runtime=RewardFunctionsRuntimeConfig(
        packages=["pandas"]
      ),
      functions={
        "output_format_check": output_format_check,
        "uses_previous_feedback": uses_previous_feedback,
        "guess_value": guess_value,
      }
    ),
    sampling_params=SamplingParamsConfig(max_tokens=4096),
    num_generations=16
  ),
  dataset=dataset,
  repo="wordle",
  description="Wordle GRPO"
)
