import uuid
import asyncio
from tqdm.asyncio import tqdm
from typing import Any, Dict, List, Optional, Tuple
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams



def init_engine(model_path: str, dtype: str, **kwargs: Any) -> AsyncLLMEngine:
  engine_args = AsyncEngineArgs(model=model_path, dtype=dtype, **kwargs)
  return AsyncLLMEngine.from_engine_args(engine_args)


async def _generate_one(
  engine: AsyncLLMEngine,
  tokenizer,
  prompt: str,
  n_samples: int = 1,
  **sampling_kwargs: Any,
) -> List[str]:
  sp = SamplingParams(
    n=n_samples,
    **sampling_kwargs,
  )

  req_id = "req-" + str(uuid.uuid4())
  generator = engine.generate(prompt, sp, req_id)

  outputs = []
  async for output in generator:
    for o in output.outputs:
      if not o.finished():
        continue
      outputs.append(o.text)

  return outputs


async def run_batch_inference(
  engine: AsyncLLMEngine,
  tokenizer,
  dataset, 
  n_samples: int,
  target_path: Optional[str] = None,
  max_concurrency: Optional[int] = None,
  **sampling_kwargs: Any,
) -> None:
  sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

  async def worker(example: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    if sem:
      async with sem:
        completions = await _generate_one(
          engine=engine,
          tokenizer=tokenizer,
          prompt=example["prompt"],
          n_samples=n_samples,
          **sampling_kwargs,
      )
    else:
      completions = await _generate_one(
        engine=engine,
        tokenizer=tokenizer,
        prompt=example["prompt"],
        n_samples=n_samples,
        **sampling_kwargs,
    )
    return example, completions

  results = []
  tasks = [asyncio.create_task(worker(e)) for e in dataset]
  for fut in tqdm.as_completed(tasks, total=len(tasks)):
    example, completions = await fut
    results.append(dict(example=example, completions=completions))
  return results
