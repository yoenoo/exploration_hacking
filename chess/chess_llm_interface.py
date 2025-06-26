from chess_validator import ChessValidator
import json

class ChessLLMInterface:
    def __init__(self):
        self.game = ChessValidator()
        self.total_reward = 0
        self.move_count = 0
        self.game_active = True
        self.move_history = []
    
    def get_board_state(self):
        """Returns the current board state in a format suitable for LLM processing"""
        board_text = []
        board_text.append("Current Board State:")
        board_text.append("  a b c d e f g h")
        board_text.append(" +----------------+")
        
        for i in range(8):
            row_num = 8 - i
            row_str = f"{row_num}|"
            
            for j in range(8):
                piece = self.game.board[i][j]
                if piece:
                    row_str += f"{self.game.piece_symbols[piece]} "
                else:
                    row_str += ". "
            
            row_str += f"|{row_num}"
            board_text.append(row_str)
        
        board_text.append(" +----------------+")
        board_text.append("  a b c d e f g h")
        
        return "\n".join(board_text)
    
    def get_game_info(self):
        """Returns comprehensive game information for LLM"""
        return {
            "board_state": self.get_board_state(),
            "current_player": self.game.current_player,
            "move_count": self.move_count,
            "total_reward": self.total_reward,
            "last_5_moves": self.move_history[-5:] if self.move_history else [],
            "game_active": self.game_active
        }
    
    def calculate_reward(self, from_pos, to_pos):
        """Calculate reward for a move"""
        target_piece = self.game.get_piece(to_pos)
        
        # Base rewards
        rewards = {
            'valid_move': 1,
            'capture_pawn': 3,
            'capture_knight': 5,
            'capture_bishop': 5,
            'capture_rook': 8,
            'capture_queen': 15,
            'developing_piece': 2,
            'center_control': 2
        }
        
        reward = rewards['valid_move']
        
        # Capture rewards
        if target_piece:
            piece_type = target_piece[1]
            if piece_type == 'P':
                reward += rewards['capture_pawn']
            elif piece_type in ['N', 'B']:
                reward += rewards['capture_knight']
            elif piece_type == 'R':
                reward += rewards['capture_rook']
            elif piece_type == 'Q':
                reward += rewards['capture_queen']
        
        # Early game development bonus
        if self.move_count < 10:
            piece = self.game.get_piece(from_pos)
            if piece and piece[1] in ['N', 'B']:
                reward += rewards['developing_piece']
        
        # Center control bonus
        to_col = ord(to_pos[0]) - ord('a')
        to_row = int(to_pos[1])
        if 2 <= to_col <= 5 and 3 <= to_row <= 6:
            reward += rewards['center_control']
        
        return reward
    
    def process_move(self, move_input):
        """Process a move from the LLM"""
        if not self.game_active:
            return {
                "success": False,
                "message": "Game is no longer active",
                "reward": 0
            }
        
        # Parse move input (expects format like "e2 e4" or "e2-e4")
        move_parts = move_input.strip().replace('-', ' ').split()
        
        if len(move_parts) != 2:
            return {
                "success": False,
                "message": "Invalid move format. Use 'from_pos to_pos' (e.g., 'e2 e4')",
                "reward": 0
            }
        
        from_pos, to_pos = move_parts[0].lower(), move_parts[1].lower()
        
        # Validate move
        is_valid, message = self.game.is_valid_move(from_pos, to_pos)
        
        if is_valid:
            # Calculate reward before making the move
            reward = self.calculate_reward(from_pos, to_pos)
            
            # Make the move
            self.game.validate_and_move(from_pos, to_pos)
            
            # Update tracking
            self.total_reward += reward
            self.move_count += 1
            self.move_history.append({
                "move": f"{from_pos}-{to_pos}",
                "player": self.game.current_player,
                "reward": reward
            })
            
            return {
                "success": True,
                "message": f"Move successful: {from_pos} to {to_pos}",
                "reward": reward,
                "total_reward": self.total_reward,
                "next_player": self.game.current_player
            }
        else:
            # Invalid move - stop the game
            self.game_active = False
            return {
                "success": False,
                "message": f"Invalid move: {message}. Game stopped.",
                "reward": -10,  # Penalty for invalid move
                "total_reward": self.total_reward - 10,
                "game_active": False
            }
    
    def reset_game(self):
        """Reset the game for a new session"""
        self.game = ChessValidator()
        self.total_reward = 0
        self.move_count = 0
        self.game_active = True
        self.move_history = []
        return "Game reset successfully"


def simulate_llm_game():
    """Simulate an LLM playing chess with the interface"""
    interface = ChessLLMInterface()
    
    print("=== Chess LLM Interface Demo ===\n")
    
    # Simulated LLM moves (some valid, one invalid at the end)
    llm_moves = [
        "e2 e4",   # White pawn opening
        "e7 e5",   # Black pawn response
        "g1 f3",   # White knight development
        "b8 c6",   # Black knight development
        "f1 c4",   # White bishop development
        "f8 c5",   # Black bishop development
        "d2 d3",   # White pawn move
        "g8 f6",   # Black knight development
        "c1 g5",   # White bishop development
        "h7 h6",   # Black pawn attack bishop
        "g5 h4",   # White bishop retreat
        "d7 d6",   # Black pawn move
        "f3 g5",   # White knight attack
        "f6 g4",   # Black knight counterattack
        "e1 e3"    # Invalid king move - will stop the game
    ]
    
    for move in llm_moves:
        print(f"\n--- Move {interface.move_count + 1} ---")
        print(interface.get_board_state())
        print(f"\nCurrent player: {interface.game.current_player}")
        print(f"Total reward so far: {interface.total_reward}")
        
        print(f"\nLLM attempts: {move}")
        result = interface.process_move(move)
        
        print(f"Result: {result['message']}")
        print(f"Reward: {result['reward']}")
        
        if not result['success']:
            print("\n!!! Game stopped due to invalid move !!!")
            break
    
    print("\n=== Final Game Statistics ===")
    print(f"Total moves played: {interface.move_count}")
    print(f"Total reward earned: {interface.total_reward}")
    print(f"Average reward per move: {interface.total_reward / max(1, interface.move_count):.2f}")
    print("\nMove history:")
    for i, move in enumerate(interface.move_history, 1):
        print(f"  {i}. {move['move']} (reward: {move['reward']})")


def interactive_llm_mode():
    """Interactive mode where you can input moves as if you were an LLM"""
    interface = ChessLLMInterface()
    
    print("=== Interactive Chess LLM Interface ===")
    print("Enter moves in format 'from_pos to_pos' (e.g., 'e2 e4')")
    print("Type 'quit' to exit, 'reset' to start new game\n")
    
    while True:
        # Display current game state
        game_info = interface.get_game_info()
        print("\n" + game_info["board_state"])
        print(f"\nCurrent player: {game_info['current_player']}")
        print(f"Move count: {game_info['move_count']}")
        print(f"Total reward: {game_info['total_reward']}")
        
        if not game_info["game_active"]:
            print("\nGame is no longer active. Type 'reset' to start a new game.")
        
        # Get input
        move_input = input("\nEnter move: ").strip()
        
        if move_input.lower() == 'quit':
            break
        elif move_input.lower() == 'reset':
            print(interface.reset_game())
            continue
        
        # Process the move
        result = interface.process_move(move_input)
        print(f"\nResult: {result['message']}")
        if 'reward' in result:
            print(f"Reward earned: {result['reward']}")
            print(f"Total reward: {result.get('total_reward', interface.total_reward)}")


if __name__ == "__main__":
    # Run the simulation
    simulate_llm_game()
    
    # Uncomment below to run interactive mode instead
    # interactive_llm_mode()