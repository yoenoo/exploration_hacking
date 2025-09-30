import json
from datasets import load_dataset
from huggingface_hub import hf_hub_download

def fetch_kernelbench_dataset(level: int):
  return load_dataset("ScalingIntelligence/KernelBench", split=f"level_{level}")

def fetch_kernelbench_sample(model: str, level: int, problem_id: int, sample_id: int = 0):
  file_path = hf_hub_download(
    repo_id="ScalingIntelligence/kernelbench-samples",
    repo_type="dataset",
    filename=f"baseline_eval/level{level}/{model}/problem_{problem_id}/sample_{sample_id}/kernel.json"
  )
  with open(file_path, 'r') as f:
    return json.load(f)


if __name__ == "__main__":
  models = ["claude-3.5-sonnet", "openai-o1"]
  for model in models:
    for level in [1, 2]:
      compiled = 0
      correct = 0
      for problem_id in range(1, 101):
        try:
          o = fetch_kernelbench_sample(model, level=level, problem_id=problem_id)
          compiled += o["eval_result"]["eval_0"]["compiled"]
          correct += o["eval_result"]["eval_0"]["correct"]
        except:
          print(f"[WARNING] Error fetching {model} for level {level} problem {problem_id}")
          continue

      print(f"{model} level {level} | compiled={compiled} | correct={correct}")