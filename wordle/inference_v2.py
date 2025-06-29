import warnings
warnings.filterwarnings("ignore", message="TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.*")

import os
import re
import ast
import torch
import pandas as pd
from tqdm.asyncio import tqdm

from transformers import AutoTokenizer
from datasets import load_dataset

import uuid
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

from dotenv import load_dotenv
load_dotenv()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, required=True)
FLAGS,_ = parser.parse_known_args()

def extract_guess(completion):
  try:
    return re.search(r'<guess>(.*?)</guess>$', completion).group(1).strip()
  except:
    return None

SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close 
your guess was.

### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example:
Secret Word: BRISK

Guess 1: STORM → Feedback: S(-) T(x) O(x) R(-) M(x)
Guess 2: BRAVE → Feedback: B(✓) R(✓) A(x) V(x) E(x)
Guess 3: BRISK → Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)

### Response Format:
Think through the problem and feedback step by step. Make sure to 
first add your step by step thought process within <think> </think> 
tags. Then, return your guessed word in the following format: 
<guess> guessed-word </guess>.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"


@dataclass
class GuessWithFeedback:
    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:
        """Returns a readable string showing the guess alongside
        its letter-by-letter feedback."""
        feedback_str = " ".join(f"{letter}({fb.value})" for letter, fb in zip(self.guess, self.feedback))
        return f"{self.guess} → Feedback: {feedback_str}"


def render_user_prompt(past_guesses: list[GuessWithFeedback]) -> str:
  """Creates a user-facing prompt that includes past guesses 
  and their feedback."""
  prompt = "Make a new 5-letter word guess."
  if past_guesses:
    prompt += "\n\nHere is some previous feedback:"
    for i, guess in enumerate(past_guesses):
      prompt += f"\nGuess {i+1}: {guess}"
  return prompt

def render_prompt(past_guesses: list[GuessWithFeedback], tokenizer):
  """Formats a full chat prompt using a system message, user 
  prompt, and assistant preamble to start the model's 
  step-by-step reasoning."""
  messages = [
    {
      "role": "system",
      "content": SYSTEM_PROMPT
    },
    {
      "role": "user",
      "content": render_user_prompt(past_guesses)
    },
    {
      "role": "assistant",
      "content": "Let me solve this step by step.\n<think>"
    }
  ]

  return tokenizer.apply_chat_template(
    messages, tokenize=False, continue_final_message=True
  )

# def next_turn(
#     past_guesses: List[GuessWithFeedback], 
#     secret_word: str, 
#     tokenizer,
#   ):
#   prompt = render_prompt(past_guesses, tokenizer)
#   completion = generate_stream(prompt)
#   match = re.search(r"<guess>\s*(.*?)\s*</guess>", completion, re.DOTALL)
#   if not match:
#     raise RuntimeError("invalid guess")
  
#   guess = match.group(1).upper()
#   feedback = get_feedback(guess, secret_word)
#   past_guesses.append(GuessWithFeedback(guess, feedback))
#   print("\n\n")
#   print(("-" * 100) + "\n")
#   for past_guess in past_guesses:
#     print(past_guess)
  
#   if guess == secret_word:
#     print("🎉 SUCCESS 🎉")
#   elif len(past_guesses) >= 6:
#     print("❌ better luck next time... ❌")

async def run_vllm_inference(engine, tokenizer, answer, past_guesses, max_tokens, temperature=1.0, top_p=0.95):
  sampling_params = SamplingParams(
    max_tokens=max_tokens,
    n=2,                      # Number of completions to sample # n=1 is much slower?
    temperature=temperature,  # Increase for more diversity
    top_p=top_p,              # Typical value for nucleus sampling
  )
  formatted_prompt = render_prompt(past_guesses, tokenizer)
  generator = engine.generate(formatted_prompt, sampling_params, uuid.uuid4())

  outputs = []
  final_output = None 
  async for output in generator:
    final_output = output 

  out = final_output.outputs[0].text
  guess = extract_guess(out)
  return dict(question=formatted_prompt, completion=out, past_guesses=past_guesses, answer=answer, guess=guess)

async def main(dataset):
  engine_args = AsyncEngineArgs(
    model=FLAGS.model_id,
    dtype="bfloat16", 
    disable_log_requests=True,
    disable_log_stats=True,
  )
  engine = AsyncLLMEngine.from_engine_args(engine_args)
  tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_id, token=os.environ["HF_TOKEN"])

  tasks = []
  for example in dataset:
    answer = example["secret"]
    _past_guesses = ast.literal_eval(example["past_guess_history"])
    past_guesses = [] 
    for guess, feedback in _past_guesses:
      fbs = []
      for fb in feedback.split(" "):
        if fb.strip()[-2] == "✓":
          fbs.append(LetterFeedback.CORRECT)
        elif fb.strip()[-2] == "-":
          fbs.append(LetterFeedback.WRONG_POS)
        elif fb.strip()[-2] == "x":
          fbs.append(LetterFeedback.WRONG_LETTER)
        else:
          raise ValueError(f"Invalid feedback: {fb}")
      past_guesses.append(GuessWithFeedback(guess, fbs))
    
    tasks.append(run_vllm_inference(engine, tokenizer, answer, past_guesses, max_tokens=4096, temperature=1.0, top_p=0.95))
    if len(tasks) > 10:
      break

  results = []
  for coroutine in tqdm.as_completed(tasks, total=len(tasks)):
    results.append(await coroutine)  

  # save results
  model_name = FLAGS.model_id.split('/')[-1]
  pd.DataFrame(results).to_csv(f"wordle_rollouts_{model_name}.csv", index=False)

if __name__ == "__main__":
  from datasets import load_dataset
  train_dataset = load_dataset("predibase/wordle-grpo", split="train")

  import asyncio
  asyncio.run(main(train_dataset))