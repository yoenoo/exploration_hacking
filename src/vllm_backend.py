import asyncio
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tqdm.asyncio import tqdm
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


THINK_TOKEN = "<think>"


def init_engine(model_path: str, dtype: str, **kwargs: Any) -> AsyncLLMEngine:
    """
    Initialize an AsyncLLMEngine with sane defaults passed via **kwargs.
    Keep env-var side effects (e.g., attention backend) OUTSIDE this module.
    """
    engine_args = AsyncEngineArgs(model=model_path, dtype=dtype, **kwargs)
    return AsyncLLMEngine.from_engine_args(engine_args)

def _format_prompt(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    use_think: bool,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Build a chat-formatted prompt for Qwen/QwQ-style templates.

    Important: If we inject an assistant message containing <think>,
    we should NOT also add a generation prompt. Otherwise, set it True.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if use_think:
        messages.append({"role": "assistant", "content": THINK_TOKEN})

    if reasoning_effort is not None:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=use_think,
            reasoning_effort=reasoning_effort,
        ).strip()
    else:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=use_think,
        ).strip()

    return prompt

async def _generate_one(
    engine: AsyncLLMEngine,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    *,
    n_samples: int = 1,
    use_think: bool = False,
    detokenize: bool = False,
    reasoning_effort: Optional[str] = None,
    **sampling_kwargs: Any,
) -> List[str]:
    """
    Generate completions for a single (system_prompt, user_prompt) pair.

    - If detokenize=False (recommended for long outputs), we collect token IDs
      while streaming and call tokenizer.batch_decode once at the end.
    - If detokenize=True, vLLM will decode each finished stream (more CPU).
    """
    sp = SamplingParams(
        n=n_samples,
        detokenize=detokenize,
        **sampling_kwargs,
    )

    prompt = _format_prompt(tokenizer, system_prompt, user_prompt, use_think, reasoning_effort)

    req_id = uuid.uuid4()
    generator = engine.generate(prompt, sp, req_id)

    finished_texts: List[str] = []
    finished_ids: List[List[int]] = []

    async for output in generator:
        for o in output.outputs:
            if not o.finished():
                continue
            if detokenize:
                finished_texts.append(o.text)
            else:
                finished_ids.append(o.token_ids)

    if not detokenize:
        # One fast batched decode instead of per-sample loops
        finished_texts = tokenizer.batch_decode(
            finished_ids,
            skip_special_tokens=True,
        )

    # Keep exactly n_samples (vLLM sometimes returns more if you change params mid-run)
    return finished_texts[:n_samples]


async def run_vllm(
    engine: AsyncLLMEngine,
    tokenizer,
    dataset: Iterable[Dict[str, Any]],
    *,
    system_prompt: str,
    max_tokens: int,
    n_samples: int,
    use_think: bool = False,
    parse_fn: Optional[Callable[[Dict[str, Any], List[str]], None]] = None,
    detokenize: bool = True,
    reasoning_effort: Optional[str] = None,
    **sampling_kwargs: Any,
) -> None:
    """
    Orchestrate generation over a dataset of examples.

    Args:
        engine: vLLM async engine.
        tokenizer: HF tokenizer.
        dataset: Iterable of dicts where each has a precomputed "prompt".
        system_prompt: Optional system message.
        max_tokens: Generation budget (passed to SamplingParams).
        n_samples: Number of samples per example.
        use_think: If True, injects an assistant message with <think>.
        parse_fn: Callback(example_dict, completions) to handle outputs.
        concurrency: Max concurrent in-flight generations (semaphore).
        detokenize: If True, let vLLM detokenize; else batch-decode at end.
        **sampling_kwargs: Any extra SamplingParams (e.g., temperature, top_p, stop).
    """
    async def worker(example: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        completions = await _generate_one(
            engine=engine,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=example["prompt"],
            n_samples=n_samples,
            use_think=use_think,
            detokenize=detokenize,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            **sampling_kwargs,
        )
        return example, completions

    tasks = [asyncio.create_task(worker(ex)) for ex in dataset]
    for fut in tqdm.as_completed(tasks, total=len(tasks)):
        example, completions = await fut
        if parse_fn is not None:
            parse_fn(example, completions)