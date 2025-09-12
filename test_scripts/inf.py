import torch
import random
import numpy as np 
import requests
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")


seed = 42
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

set_seed(seed)



BASE = "http://127.0.0.1:8000" 

prompt = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is 1+2?"},
  # {"role": "assistant", "content": ""}
]
print(prompt)
prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
print(prompt)
print("="*100)


# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B")

# gc = model.generation_config
# gc.temperature = None
# gc.top_p = None
# gc.top_k = None

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
# inputs = {k: v.to(model.device) for k, v in inputs.items()}
# token_ids = model.generate(
#   **inputs,
#   max_new_tokens=100,
#   do_sample=False,
#   # pad_token_id=tokenizer.pad_token_id,
# )
# decoded_text = tokenizer.batch_decode(token_ids)
# print(decoded_text)


from vllm import SamplingParams

def generate_completion(tokenizer, prompt, max_tokens, **kwargs):
  if isinstance(prompt, list):
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

  payload = {
    "prompts": [prompt],
    "generation_kwargs": {
      "seed": seed,
      "max_tokens": max_tokens,
      **kwargs,
    }
  }

  resp = requests.post(f"{BASE}/generate", json=payload)
  resp.raise_for_status()
  token_ids = resp.json()["completion_ids"]
  decoded_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
  return decoded_text


print(generate_completion(tokenizer, "You are a helpful assistant.", "What is 1+2?", 100, temperature=0))