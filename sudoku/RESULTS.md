# 3x3 Sudoku solver

| Model | Metric | Difficulty | Score | 
| --- | --- | --- | --- | 
| 3.5 Haiku | Accuracy | 0.1 | 0.7 |
| 3.5 Haiku | Accuracy | 0.2 | 0.3 |
| 3.5 Haiku | Accuracy | 0.3 | 0.4 |
| 3.5 Haiku | Accuracy | 0.5 | 0.0 |
| 3.5 Haiku | Accuracy | 0.9 | 0.0 |
| 3.5 Haiku | % Correct | 0.1 | 0.9375 | 
| 3.5 Haiku | % Correct | 0.2 | 0.7875 | 
| 3.5 Haiku | % Correct | 0.3 | 0.8250 | 
| 3.5 Haiku | % Correct | 0.5 | 0.4600 | 
| 3.5 Haiku | % Correct | 0.9 | 0.1306 | 
| 4 Sonnet | Accuracy | 0.1 | 1.0 |
| 4 Sonnet | Accuracy | 0.2 | 0.9 |
| 4 Sonnet | Accuracy | 0.3 | 0.3 |
| 4 Sonnet | Accuracy | 0.5 | 0.0 |
| 4 Sonnet | Accuracy | 0.9 | 0.0 |
| 4 Sonnet | % Correct | 0.1 | 1.0 | 
| 4 Sonnet | % Correct | 0.2 | 0.9375 | 
| 4 Sonnet | % Correct | 0.3 | 0.8458 | 
| 4 Sonnet | % Correct | 0.5 | 0.6300 | 
| 4 Sonnet | % Correct | 0.9 | 0.1833 | 


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