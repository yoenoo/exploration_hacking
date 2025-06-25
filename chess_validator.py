from chess_environment import ChessBoard

class ChessValidator(ChessBoard):
    def __init__(self):
        super().__init__()
    
    def is_valid_move(self, from_pos, to_pos):
        piece = self.get_piece(from_pos)
        if not piece:
            return False, "No piece at starting position"
        
        if piece[0] == 'w' and self.current_player != 'white':
            return False, "It's black's turn"
        if piece[0] == 'b' and self.current_player != 'black':
            return False, "It's white's turn"
        
        target_piece = self.get_piece(to_pos)
        if target_piece and target_piece[0] == piece[0]:
            return False, "Cannot capture your own piece"
        
        from_col = ord(from_pos[0]) - ord('a')
        from_row = 8 - int(from_pos[1])
        to_col = ord(to_pos[0]) - ord('a')
        to_row = 8 - int(to_pos[1])
        
        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            return False, "Target position is out of bounds"
        
        piece_type = piece[1]
        
        if piece_type == 'P':
            return self.is_valid_pawn_move(from_row, from_col, to_row, to_col, piece[0])
        elif piece_type == 'R':
            return self.is_valid_rook_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'N':
            return self.is_valid_knight_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'B':
            return self.is_valid_bishop_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'Q':
            return self.is_valid_queen_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'K':
            return self.is_valid_king_move(from_row, from_col, to_row, to_col)
        
        return False, "Unknown piece type"
    
    def is_valid_pawn_move(self, from_row, from_col, to_row, to_col, color):
        direction = -1 if color == 'w' else 1
        start_row = 6 if color == 'w' else 1
        
        if from_col == to_col:
            if to_row == from_row + direction and not self.board[to_row][to_col]:
                return True, "Valid pawn move"
            if from_row == start_row and to_row == from_row + 2 * direction:
                if not self.board[to_row][to_col] and not self.board[from_row + direction][from_col]:
                    return True, "Valid pawn double move"
        
        elif abs(from_col - to_col) == 1 and to_row == from_row + direction:
            if self.board[to_row][to_col] and self.board[to_row][to_col][0] != color:
                return True, "Valid pawn capture"
        
        return False, "Invalid pawn move"
    
    def is_valid_rook_move(self, from_row, from_col, to_row, to_col):
        if from_row != to_row and from_col != to_col:
            return False, "Rook must move in straight lines"
        
        if from_row == to_row:
            step = 1 if to_col > from_col else -1
            for col in range(from_col + step, to_col, step):
                if self.board[from_row][col]:
                    return False, "Path is blocked"
        else:
            step = 1 if to_row > from_row else -1
            for row in range(from_row + step, to_row, step):
                if self.board[row][from_col]:
                    return False, "Path is blocked"
        
        return True, "Valid rook move"
    
    def is_valid_knight_move(self, from_row, from_col, to_row, to_col):
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        if (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2):
            return True, "Valid knight move"
        
        return False, "Invalid knight move"
    
    def is_valid_bishop_move(self, from_row, from_col, to_row, to_col):
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        if row_diff != col_diff:
            return False, "Bishop must move diagonally"
        
        row_step = 1 if to_row > from_row else -1
        col_step = 1 if to_col > from_col else -1
        
        row, col = from_row + row_step, from_col + col_step
        while row != to_row:
            if self.board[row][col]:
                return False, "Path is blocked"
            row += row_step
            col += col_step
        
        return True, "Valid bishop move"
    
    def is_valid_queen_move(self, from_row, from_col, to_row, to_col):
        is_rook_move = self.is_valid_rook_move(from_row, from_col, to_row, to_col)
        if is_rook_move[0]:
            return True, "Valid queen move"
        
        is_bishop_move = self.is_valid_bishop_move(from_row, from_col, to_row, to_col)
        if is_bishop_move[0]:
            return True, "Valid queen move"
        
        return False, "Invalid queen move"
    
    def is_valid_king_move(self, from_row, from_col, to_row, to_col):
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        if row_diff <= 1 and col_diff <= 1:
            return True, "Valid king move"
        
        return False, "Invalid king move"
    
    def validate_and_move(self, from_pos, to_pos):
        is_valid, message = self.is_valid_move(from_pos, to_pos)
        
        if is_valid:
            self.move_piece(from_pos, to_pos)
        
        return is_valid, message


def main():
    game = ChessValidator()
    
    print("Chess Game with Move Validation")
    print("================================\n")
    
    game.display()
    
    moves = [
        ("e2", "e4", "Valid opening move"),
        ("e7", "e5", "Black's response"),
        ("g1", "f3", "Knight development"),
        ("b8", "c6", "Black knight out"),
        ("f1", "c4", "Bishop development"),
        ("f7", "f6", "Weak pawn move"),
        ("f3", "e5", "Knight takes pawn"),
        ("f6", "e5", "Pawn recaptures"),
        ("d1", "h5", "Queen attack!"),
        ("c8", "e6", "Bishop development")
    ]
    
    for from_pos, to_pos, description in moves:
        print(f"\n{description}: {from_pos} to {to_pos}")
        is_valid, message = game.validate_and_move(from_pos, to_pos)
        print(f"Result: {message}")
        
        if is_valid:
            print(f"Current player: {game.current_player}")
            game.display()
        else:
            print("Move rejected!")
    
    print("\n\nTesting invalid moves:")
    print("======================")
    
    invalid_moves = [
        ("e2", "e5", "Pawn trying to move 3 squares"),
        ("a1", "a3", "Rook trying to jump over pieces"),
        ("c1", "e3", "Bishop blocked by pawns"),
        ("e1", "e3", "King moving 2 squares"),
    ]
    
    test_game = ChessValidator()
    for from_pos, to_pos, description in invalid_moves:
        print(f"\n{description}: {from_pos} to {to_pos}")
        is_valid, message = test_game.is_valid_move(from_pos, to_pos)
        print(f"Result: {message}")


if __name__ == "__main__":
    main()