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

def calculate_feedback(guess: str, secret: str) -> List[LetterFeedback]:
  """Calculate Wordle feedback for a guess against the secret word."""
  if len(secret) != 5:
    raise ValueError(f"Secret word must be 5 letters, got {len(secret)}")
  
  # Handle guesses that aren't 5 letters
  guess_upper = guess.upper()
  if len(guess_upper) < 5:
    # Pad with spaces to make it 5 characters
    guess_upper = guess_upper.ljust(5)
  elif len(guess_upper) > 5:
    # Truncate to 5 characters
    guess_upper = guess_upper[:5]
  
  feedback = []
  secret_chars = list(secret.upper())
  guess_chars = list(guess_upper)
  
  # First pass: mark correct positions
  for i in range(5):
    if guess_chars[i] == ' ':
      # Padding character is always wrong
      feedback.append(LetterFeedback.WRONG_LETTER)
    elif guess_chars[i] == secret_chars[i]:
      feedback.append(LetterFeedback.CORRECT)
      secret_chars[i] = None  # Mark as used
    else:
      feedback.append(None)
  
  # Second pass: mark wrong positions
  for i in range(5):
    if feedback[i] is None:
      if guess_chars[i] in secret_chars:
        feedback[i] = LetterFeedback.WRONG_POS
        # Remove first occurrence of this letter
        secret_chars[secret_chars.index(guess_chars[i])] = None
      else:
        feedback[i] = LetterFeedback.WRONG_LETTER
  
  return feedback

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

def play_wordle_game(secret_word: str, model_id: str, max_attempts: int = 5) -> dict:
  """
  Play a multi-turn Wordle game with Claude.
  
  Args:
    secret_word: The 5-letter word to guess
    model_id: The Claude model to use
    max_attempts: Maximum number of guesses allowed (default: 5)
    
  Returns:
    Dictionary containing game history and result
  """
  past_guesses = []
  game_history = []
  
  for attempt in range(max_attempts):
    # Generate prompt with past guesses
    messages = render_prompt(past_guesses)
    
    # Get Claude's response
    completion = run_inference(model_id=model_id, messages=messages, max_tokens=4096, temperature=0)
    
    # Extract the guess
    guess = extract_guess(completion)
    if not guess:
      print(f"Failed to extract guess from completion: {completion}")
      return 
    
    # Calculate feedback
    feedback = calculate_feedback(guess, secret_word)
    guess_with_feedback = GuessWithFeedback(guess, feedback)
    past_guesses.append(guess_with_feedback)
    
    # Record this turn
    game_history.append({
      "attempt": attempt + 1,
      "guess": guess,
      "feedback": guess_with_feedback.__repr__(),
      "completion": completion,
      "correct": guess.upper() == secret_word.upper()
    })
    
    # Check if won
    if guess.upper() == secret_word.upper():
      print(f"🎉 Claude guessed correctly in {attempt + 1} attempts!")
      return {
        "success": True,
        "attempts": attempt + 1,
        "secret_word": secret_word,
        "history": game_history
      }
    
    print(f"Attempt {attempt + 1}: {guess_with_feedback}")
  
  print(f"❌ Claude failed to guess '{secret_word}' in {max_attempts} attempts")
  return {
    "success": False,
    "attempts": max_attempts,
    "secret_word": secret_word,
    "history": game_history
  }

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, required=True)
  args = parser.parse_args()
  

  print("=== Example 1: Multiple Games ===")
  test_words = [
    "CRATE", "SPARK", "PLANT", "MOUSE", "BEACH", 
    "BRAIN", "MANGO", "ALLOY", "CREED", "THIEF"
  ]
  
  for word in test_words:
    print(f"\n--- Testing word: {word} ---")
    result = play_wordle_game(word, model_id=args.model)