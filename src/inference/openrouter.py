import os
import asyncio
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pathlib import Path

env_path = ".env"
load_dotenv(env_path, override=True)


class OpenRouterClient:
  def __init__(self, api_key=None):
    _base_url = "https://openrouter.ai/api/v1"
    _api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    self.client = AsyncOpenAI(base_url=_base_url, api_key=_api_key)
    self.global_semaphore = asyncio.Semaphore(100)  # Global limit on concurrent requests
  
  async def generate(self, model: str, messages: list[dict], n_rollouts: int, **kwargs):
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests per prompt
    
    async def generate_with_semaphore():
      # retries = 10
      async with semaphore:
        async with self.global_semaphore:  # Also respect global limit
          try:
            response = await self.client.chat.completions.create(
              model=model,
              messages=messages,
              **kwargs,
            )
            return response.choices[0].message.content
          except Exception as e:
            print(e)
    
    tasks = [generate_with_semaphore() for _ in range(n_rollouts)]
    responses = await asyncio.gather(*tasks)
    return responses

  async def batch_generate(self, model: str, messages: list[list[dict]], n_rollouts: int, **kwargs):
    batch_semaphore = asyncio.Semaphore(30)  # Process max 30 prompts concurrently
    
    async def batch_generate_with_index(i, msgs):
      async with batch_semaphore:
        return (i, await self.generate(model, msgs, n_rollouts=n_rollouts, **kwargs))

    tasks = [batch_generate_with_index(i, msgs) for i, msgs in enumerate(messages)]

    completions = []
    for task in tqdm.as_completed(tasks, desc=f"Generating rollouts (n={n_rollouts})", total=len(messages)):
      completions.append(await task)
    return completions



async def main():
  client = OpenRouterClient()

  model = "qwen/qwen3-235b-a22b-thinking-2507"
  messages = [[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ],]

  batch_completions = await client.batch_generate(
    model=model,
    messages=messages,
    n_rollouts=1,
    temperature=0.0,
    max_tokens=1024,
  )
  print(batch_completions)
  print(len(batch_completions))


if __name__ == "__main__":
  import asyncio
  asyncio.run(main())
