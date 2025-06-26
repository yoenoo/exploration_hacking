# 3x3 Sudoku solver

# TODO
- [ ] replicate results w/ openai and gemini
- [ ] replicate GRPO training on 2x2 sudoku problems
  - link: https://github.com/Asad-Shahab/sudokuLLM/tree/main
  - need 100GB RAM?
  - [ ] check performance on gemma (or other open weight models) -- check the inference time on consumer GPUs
  - [ ] create a sudoku dataset -- remove duplicates, 100 datasets per difficulty level (also 1 missing and only 1)
  - [ ] GRPO training using trl library (vanila)
  - [ ] improve using unsloth -- tricky
- [x] run inferece on difficulty > 0.5
- [ ] explore different prompting stuff - either close to zero shot or improve scores (more reasoning)


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


<!-- 
| 3.5 Haiku | Accuracy | 0.02 | 0.8 | o | 
| 3.5 Haiku | Accuracy | 0.1 | 0.1 | o | 
| 3.5 Haiku | Accuracy | 0.2 | 0.4 | o | 
| 3.5 Haiku | Accuracy | 0.3 | 0.0 | o | 
| 3.5 Haiku | Accuracy | 0.5 | 0.0 | o | 
| 3.5 Haiku | Accuracy | 0.9 | 0.0 | o | 
| 3.5 Haiku | % Correct | 0.02 | 0.8 | o | 
| 3.5 Haiku | % Correct | 0.1 | 0.7250 | o | 
| 3.5 Haiku | % Correct | 0.2 | 0.8375 | o | 
| 3.5 Haiku | % Correct | 0.3 | 0.6292 | o | 
| 3.5 Haiku | % Correct | 0.5 | 0.4400 | o | 
| 3.5 Haiku | % Correct | 0.9 | 0.1361 | o |  -->


- model
- sudoku difficulty 
- temperature / top_p
- number of rollouts per question
  - e.g. 10 fresh questions and 10 rollouts per question (nucleus sampling w/ top_p=0.95) -> diversity

- binary accuracy
- % cells correct


Sudoku Difficulty: 0.1
Accuracy: 0.7000
0.9375
Sudoku Difficulty: 0.2
Accuracy: 0.3000
0.7875
Sudoku Difficulty: 0.3
Accuracy: 0.4000
0.825
Sudoku Difficulty: 0.5
Accuracy: 0.0000
0.4600000000000001
Sudoku Difficulty: 0.9
Accuracy: 0.1000
0.13055555555555556


Sudoku Difficulty: 0.1
Accuracy: 1.0000
1.0
Sudoku Difficulty: 0.2
Accuracy: 0.9000
0.9375
Sudoku Difficulty: 0.3
Accuracy: 0.3000
0.8458333333333332
Sudoku Difficulty: 0.5
Accuracy: 0.0000
0.63
Sudoku Difficulty: 0.9
Accuracy: 0.0000
0.18333333333333332