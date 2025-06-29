import os
import re
import ast
import anthropic
import pandas as pd 
from tqdm import tqdm
from enum import Enum
from typing import List
from datasets import load_dataset
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()


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



class LetterFeedback(Enum):
  CORRECT = "✓"
  WRONG_POS = "-"
  WRONG_LETTER = "x"

@dataclass
class GuessWithFeedback:
  guess: str
  feedback: List[LetterFeedback]

  def __repr__(self) -> str:
    feedback_str = " ".join(f"{letter}({fb.value})" for letter, fb in zip(self.guess, self.feedback))
    return f"{self.guess} → Feedback: {feedback_str}"

def render_user_prompt(past_guesses: list[GuessWithFeedback]) -> str:
  prompt = "Make a new 5-letter word guess."
  if past_guesses:
    prompt += "\n\nHere is some previous feedback:"
    for i, guess in enumerate(past_guesses):
      prompt += f"\nGuess {i+1}: {guess}"
  return prompt

def render_prompt(past_guesses: list[GuessWithFeedback]):
  messages = [
    # {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": render_user_prompt(past_guesses)},
    {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
  ]
  return messages

def extract_guess(completion):
  try:
    return re.search(r'<guess>(.*?)</guess>$', completion).group(1).strip().upper()
  except:
    return None

def run_inference(model_id: str, messages: list[dict], max_tokens: int, temperature: float = 0.0, **kwargs):
  client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
  message = client.messages.create(
    model=model_id,
    max_tokens=max_tokens,
    temperature=temperature,
    system=SYSTEM_PROMPT,
    messages=messages,
    **kwargs,
  )
  return message.content[0].text

if __name__ == "__main__":
  dataset = load_dataset("predibase/wordle-grpo", split="train")

  results = []
  for example in tqdm(dataset):
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

    messages = render_prompt(past_guesses)
    completions = run_inference(model_id="claude-sonnet-4-20250514", messages=messages, max_tokens=4096, temperature=0)
    guess = extract_guess(completions)
    correct = guess == answer
    results.append(dict(completions=completions, past_guesses=past_guesses, guess=guess, answer=answer, correct=correct))

    if len(results) > 10:
      break


  results = pd.DataFrame(results)
  results.to_csv("wordle_claude.csv", index=False)