import asyncio
from tqdm import tqdm
from src.inference.openrouter import OpenRouterClient
from src.oxen_rustbench.dataset import create_dataset
from src.oxen_rustbench.prompt import SYSTEM_PROMPT
from src.oxen_rustbench.rewards import RustCodeEvaluator, RewardFunctions


evaluator = RustCodeEvaluator()
reward_functions = RewardFunctions(evaluator)
dataset = create_dataset(mode="train", system_prompt=SYSTEM_PROMPT)

# client = OpenRouterClient(model="qwen/qwen3-14b", temperature=0.0, max_tokens=8192)

# messages = [item["prompt"] for item in dataset][:10]
# print(len(messages))
# print(messages[0])

# rollouts = asyncio.run(client.generate(messages))
# for rollout in rollouts:
#   rollout = [r["reasoning"] + "\n\n" + r["content"] for r in rollout]
#   reward = reward_functions.cargo_test_reward(rollout)
#   print(reward)


from src.vllm_engine import run_batch_inference, init_engine
engine = init_engine("Qwen/Qwen3-14B", dtype="bfloat16")
samples = asyncio.run(run_batch_inference(
  engine,
  tokenizer,
  dataset.select(range(3)),
  n_samples=1,
))
print(samples)