import random
from validator import is_valid_sudoku
from sudoku import Sudoku


for _ in range(1000):
  difficulty = random.random()
  p = Sudoku(3).difficulty(difficulty)
  # p.show()

  solution = p.solve()
  # solution.show()

  assert is_valid_sudoku(p.board) is False
  assert is_valid_sudoku(solution.board) is True