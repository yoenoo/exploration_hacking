class ChessBoard:
    def __init__(self):
        self.board = self.initialize_board()
        self.current_player = 'white'
        
        self.piece_symbols = {
            'wK': '♔', 'wQ': '♕', 'wR': '♖', 'wB': '♗', 'wN': '♘', 'wP': '♙',
            'bK': '♚', 'bQ': '♛', 'bR': '♜', 'bB': '♝', 'bN': '♞', 'bP': '♟'
        }
    
    def initialize_board(self):
        board = [[None for _ in range(8)] for _ in range(8)]
        
        piece_order = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        
        for i in range(8):
            board[0][i] = f'b{piece_order[i]}'
            board[1][i] = 'bP'
            board[6][i] = 'wP'
            board[7][i] = f'w{piece_order[i]}'
        
        return board
    
    def display(self):
        print('  A B C D E F G H')
        print(' +----------------+')
        
        for i in range(8):
            row_num = 8 - i
            print(f'{row_num}|', end='')
            
            for j in range(8):
                if self.board[i][j]:
                    print(f'{self.piece_symbols[self.board[i][j]]} ', end='')
                else:
                    print('. ', end='')
            
            print(f'|{row_num}')
        
        print(' +----------------+')
        print('  A B C D E F G H')
    
    def get_piece(self, position):
        col = ord(position[0]) - ord('a')
        row = 8 - int(position[1])
        
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None
    
    def set_piece(self, position, piece):
        col = ord(position[0]) - ord('a')
        row = 8 - int(position[1])
        
        if 0 <= row < 8 and 0 <= col < 8:
            self.board[row][col] = piece
            return True
        return False
    
    def move_piece(self, from_pos, to_pos):
        piece = self.get_piece(from_pos)
        if piece:
            self.set_piece(to_pos, piece)
            self.set_piece(from_pos, None)
            self.current_player = 'black' if self.current_player == 'white' else 'white'
            return True
        return False


def main():
    chess = ChessBoard()
    
    print("Chess Environment Initialized!")
    print(f"Current player: {chess.current_player}")
    print()
    
    chess.display()
    
    print("\nExample moves (not validated):")
    print("Moving e2 to e4...")
    chess.move_piece('e2', 'e4')
    print(f"Current player: {chess.current_player}")
    print()
    chess.display()
    
    print("\nMoving e7 to e5...")
    chess.move_piece('e7', 'e5')
    print(f"Current player: {chess.current_player}")
    print()
    chess.display()


if __name__ == "__main__":
    main()