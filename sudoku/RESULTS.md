# 3x3 Sudoku solver

# TODO
- [ ] replicate GRPO training on 2x2 sudoku problems
  - link: https://github.com/Asad-Shahab/sudokuLLM/tree/main
  - need 100GB RAM?
  - [x] check performance on gemma (or other open weight models) -- check the inference time on consumer GPUs
  - [x] create a sudoku dataset -- remove duplicates, 100 datasets per difficulty level (also 1 missing and only 1)
    - created at: https://huggingface.co/datasets/yoenoo/sudoku
    - [ ] should do train/dev/test and generate a lot more?
  - [ ] GRPO training using trl library (vanila)
  - [ ] improve using unsloth -- tricky
  - [ ] perhaps just SFT on top 10% rollouts?
- [x] run inferece on difficulty > 0.5
- [ ] explore different prompting stuff - either close to zero shot or improve scores (more reasoning)
- [x] replicate results w/ openai 
- [ ] replicate results w/ gemini (don't have access as of now)

"""
# TODO: 
- RL simulation using ICL? SFT? already in the prompt the model is taught the rules of sudoku, but not given any examples
- malign init?
- rollouts (w/ nucleus sampling)
- can do you an RL training loop?
"""

| Model | Metric | Difficulty | Score | "RL" | 
| --- | --- | --- | --- | --- | 
| 3.5 Haiku | Accuracy | 0.02 | 0.9 | x | 
| 3.5 Haiku | Accuracy | 0.1 | 0.7 | x | 
| 3.5 Haiku | Accuracy | 0.2 | 0.3 | x | 
| 3.5 Haiku | Accuracy | 0.3 | 0.4 | x | 
| 3.5 Haiku | Accuracy | 0.4 | 0.1 | x | 
| 3.5 Haiku | Accuracy | 0.5 | 0.0 | x | 
| 3.5 Haiku | Accuracy | 0.6 | 0.0 | x | 
| 3.5 Haiku | Accuracy | 0.7 | 0.0 | x | 
| 3.5 Haiku | Accuracy | 0.8 | 0.0 | x | 
| 3.5 Haiku | Accuracy | 0.9 | 0.0 | x | 
| 3.5 Haiku | % Correct | 0.02 | 0.9 | x | 
| 3.5 Haiku | % Correct | 0.1 | 0.9375 | x | 
| 3.5 Haiku | % Correct | 0.2 | 0.7875 | x | 
| 3.5 Haiku | % Correct | 0.3 | 0.8250 | x | 
| 3.5 Haiku | % Correct | 0.4 | 0.6750 | x | 
| 3.5 Haiku | % Correct | 0.5 | 0.4600 | x | 
| 3.5 Haiku | % Correct | 0.6 | 0.3687 | x | 
| 3.5 Haiku | % Correct | 0.7 | 0.1857 | x | 
| 3.5 Haiku | % Correct | 0.8 | 0.1250 | x | 
| 3.5 Haiku | % Correct | 0.9 | 0.1306 | x | 
| 4 Sonnet | Accuracy | 0.02 | 1.0 | x |
| 4 Sonnet | Accuracy | 0.1 | 1.0 | x |
| 4 Sonnet | Accuracy | 0.2 | 0.9 | x | 
| 4 Sonnet | Accuracy | 0.3 | 0.3 | x | 
| 4 Sonnet | Accuracy | 0.4 | 0.0 | x | 
| 4 Sonnet | Accuracy | 0.5 | 0.0 | x | 
| 4 Sonnet | Accuracy | 0.6 | 0.0 | x | 
| 4 Sonnet | Accuracy | 0.7 | 0.0 | x | 
| 4 Sonnet | Accuracy | 0.8 | 0.0 | x | 
| 4 Sonnet | Accuracy | 0.9 | 0.0 | x | 
| 4 Sonnet | % Correct | 0.02 | 1.0 | x | 
| 4 Sonnet | % Correct | 0.1 | 1.0 | x | 
| 4 Sonnet | % Correct | 0.2 | 0.9375 | x | 
| 4 Sonnet | % Correct | 0.3 | 0.8458 | x | 
| 4 Sonnet | % Correct | 0.4 | 0.7125 | x | 
| 4 Sonnet | % Correct | 0.5 | 0.6300 | x | 
| 4 Sonnet | % Correct | 0.6 | 0.3958 | x | 
| 4 Sonnet | % Correct | 0.7 | 0.2946 | x | 
| 4 Sonnet | % Correct | 0.8 | 0.2156 | x | 
| 4 Sonnet | % Correct | 0.9 | 0.1833 | x | 
| Gemma-3-4B | Accuracy | 0.02 | 0.6 | x |
| Gemma-3-4B | Accuracy | 0.1 | 0.4 | x |
| Gemma-3-4B | Accuracy | 0.2 | 0.0 | x | 
| Gemma-3-4B | Accuracy | 0.3 | 0.0 | x | 
| Gemma-3-4B | Accuracy | 0.4 | 0.0 | x | 
| Gemma-3-4B | Accuracy | 0.5 | 0.0 | x | 
| Gemma-3-4B | Accuracy | 0.6 | 0.0 | x | 
| Gemma-3-4B | Accuracy | 0.7 | 0.0 | x | 
| Gemma-3-4B | Accuracy | 0.8 | 0.0 | x | 
| Gemma-3-4B | Accuracy | 0.9 | 0.0 | x | 
| Gemma-3-4B | % Correct | 0.02 | 0.6 | x | 
| Gemma-3-4B | % Correct | 0.1 | 0.6625 | x | 
| Gemma-3-4B | % Correct | 0.2 | 0.6125 | x | 
| Gemma-3-4B | % Correct | 0.3 | 0.4542 | x | 
| Gemma-3-4B | % Correct | 0.4 | 0.4188 | x | 
| Gemma-3-4B | % Correct | 0.5 | 0.3175 | x | 
| Gemma-3-4B | % Correct | 0.6 | 0.2458 | x | 
| Gemma-3-4B | % Correct | 0.7 | 0.2018 | x | 
| Gemma-3-4B | % Correct | 0.8 | 0.1828 | x | 
| Gemma-3-4B | % Correct | 0.9 | 0.1417 | x | 
| GPT-4o-mini | Accuracy | 0.02 | 0.9 | x |
| GPT-4o-mini | Accuracy | 0.1 | 0.5 | x |
| GPT-4o-mini | Accuracy | 0.2 | 0.2 | x | 
| GPT-4o-mini | Accuracy | 0.3 | 0.1 | x | 
| GPT-4o-mini | Accuracy | 0.4 | 0.0 | x | 
| GPT-4o-mini | Accuracy | 0.5 | 0.0 | x | 
| GPT-4o-mini | Accuracy | 0.6 | 0.0 | x | 
| GPT-4o-mini | Accuracy | 0.7 | 0.0 | x | 
| GPT-4o-mini | Accuracy | 0.8 | 0.0 | x | 
| GPT-4o-mini | Accuracy | 0.9 | 0.0 | x | 
| GPT-4o-mini | % Correct | 0.02 | 1.0 | x | 
| GPT-4o-mini | % Correct | 0.1 | 0.8125 | x | 
| GPT-4o-mini | % Correct | 0.2 | 0.6750 | x | 
| GPT-4o-mini | % Correct | 0.3 | 0.7375 | x | 
| GPT-4o-mini | % Correct | 0.4 | 0.5 | x | 
| GPT-4o-mini | % Correct | 0.5 | 0.3325 | x | 
| GPT-4o-mini | % Correct | 0.6 | 0.3187 | x | 
| GPT-4o-mini | % Correct | 0.7 | 0.2071 | x | 
| GPT-4o-mini | % Correct | 0.8 | 0.1922 | x | 
| GPT-4o-mini | % Correct | 0.9 | 0.15 | x | 

| o3 | Accuracy | 0.02 | 1.0 | x |
| o3 | Accuracy | 0.1 | 1.0 | x |
| o3 | Accuracy | 0.2 | 1.0 | x | 
| o3 | Accuracy | 0.3 | 1.0 | x | 
| o3 | Accuracy | 0.4 | 1.0 | x | 
| o3 | Accuracy | 0.5 | 1.0 | x | 
| o3 | Accuracy | 0.6 | 1.0 | x | 
| o3 | Accuracy | 0.7 | 0.0 | x | 
| o3 | Accuracy | 0.8 | 0.0 | x | 
| o3 | Accuracy | 0.9 | 0.0 | x | 
| o3 | % Correct | 0.02 | 1.0 | x | 
| o3 | % Correct | 0.1 | 1.0 | x | 
| o3 | % Correct | 0.2 | 1.0 | x | 
| o3 | % Correct | 0.3 | 1.0 | x | 
| o3 | % Correct | 0.4 | 0.9719 | x | 
| o3 | % Correct | 0.5 | 0.9175 | x | 
| o3 | % Correct | 0.6 | 0.8646 | x | 

| o3 | % Correct | 0.7 | 0.2071 | x | 
| o3 | % Correct | 0.8 | 0.1922 | x | 
| o3 | % Correct | 0.9 | 0.15 | x | 


o3 does a lot of thinking..
% correct can be inflated/deflated because there could be multiple solutions to a problem..