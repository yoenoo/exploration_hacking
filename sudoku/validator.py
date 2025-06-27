def is_valid_sudoku(board, verbose=False):
    """
    Check if a solved Sudoku board is valid.
    
    Args:
        board: 9x9 list of lists representing the Sudoku board
               Each cell must contain a number 1-9 (no empty cells allowed)
    
    Returns:
        bool: True if the Sudoku is valid and complete, False otherwise
    """
    
    # Check if board is 9x9
    if len(board) != 9 or any(len(row) != 9 for row in board):
        if verbose: print("❌ Board must be 9x9")
        return False
    
    # Check if all cells are filled with numbers 1-9 (no empty cells)
    for i in range(9):
        for j in range(9):
            if not isinstance(board[i][j], int) or board[i][j] < 1 or board[i][j] > 9:
                if verbose: print(f"❌ Empty or invalid cell at position ({i}, {j}): {board[i][j]}")
                return False
    
    # Track numbers in rows, columns, and 3x3 sub-grids
    rows = [[0] * 10 for _ in range(9)]  # rows[i][val] = count of val in row i
    cols = [[0] * 10 for _ in range(9)]  # cols[j][val] = count of val in col j
    subgrids = [[0] * 10 for _ in range(9)]  # subgrids[k][val] = count of val in subgrid k
    
    for i in range(9):
        for j in range(9):
            val = board[i][j]
            
            # Check row
            if rows[i][val] > 0:
                if verbose: print(f"❌ Duplicate {val} in row {i}")
                return False
            rows[i][val] = 1
            
            # Check column
            if cols[j][val] > 0:
                if verbose: print(f"❌ Duplicate {val} in column {j}")
                return False
            cols[j][val] = 1
            
            # Check 3x3 sub-grid
            subgrid_idx = (i // 3) * 3 + (j // 3)
            if subgrids[subgrid_idx][val] > 0:
                if verbose: print(f"❌ Duplicate {val} in sub-grid {subgrid_idx}")
                return False
            subgrids[subgrid_idx][val] = 1
    
    if verbose: print("✅ Sudoku is valid and complete!")
    return True