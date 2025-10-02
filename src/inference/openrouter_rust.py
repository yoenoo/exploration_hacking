import os
import httpx
import asyncio
from dotenv import load_dotenv
load_dotenv()


class OpenRouterClient:
  def __init__(self, model: str, max_concurrent: int = 10, **sampling_kwargs):
    self.url = "https://openrouter.ai/api/v1/chat/completions"

    self.api_key = os.environ.get("OPENROUTER_API_KEY")
    if not self.api_key:
      raise ValueError("OPENROUTER_API_KEY not found in environment or provided explicitly.")

    self._headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
    self._payload = {
      "model": model,
      **sampling_kwargs,
    }
    self.semaphore = asyncio.Semaphore(max_concurrent)

  async def get_completion(self, client: httpx.AsyncClient, messages: list[dict[str]], task_id: str):
    async with self.semaphore:
      self._payload.update({"messages": messages})
      response = await client.post(self.url, headers=self._headers, json=self._payload)
      return task_id, response.json()["choices"][0]["message"]

  async def generate(self, messages: list[list[dict[str]]]):
    async with httpx.AsyncClient() as client:
      tasks = [self.get_completion(client, msg) for msg in messages]
      task_ids, results = await asyncio.gather(*tasks)
    return task_ids, results

  async def generate_rollouts(self, messages: list[dict[str]], n: int = 16):
    async with httpx.AsyncClient() as client:
      tasks = [self.get_completion(client, messages) for i in range(n)]
      results = await asyncio.gather(*tasks)
    return results



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, required=True)
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_p", type=float, default=0.95)
  parser.add_argument("--max_tokens", type=int, default=128)
  parser.add_argument("-n", "--n_rollouts", type=int, default=1) 
  args = parser.parse_args()

  client = OpenRouterClient(model=args.model, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "what is 2+123?"}
  ]
  import time 

  start_time = time.perf_counter()
  rollouts = asyncio.run(client.generate_rollouts(messages, n=args.n_rollouts))
  for rollout in rollouts:
    print(rollout)

  end_time = time.perf_counter()
  print(f"Time taken: {end_time - start_time:.3f}s")