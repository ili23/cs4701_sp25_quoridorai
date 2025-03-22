from game_state_tuple import (
    create_initial_state, 
    apply_move, 
    get_possible_moves, 
    is_terminal,
    print_game_state,
    check_winner
)

from MCTS import Agent
import random

def game_loop(player0_is_bot=False, player1_is_bot=False):
    """
    Run a game of Quoridor with specified player types.
    
    Parameters:
        player0_is_bot (bool): Whether player 0 is a bot (True) or human (False)
        player1_is_bot (bool): Whether player 1 is a bot (True) or human (False)
    
    Returns:
        int: The winner of the game (0 or 1)
    """
    state = create_initial_state()
    print_game_state(state)
    
    # Keep track of player types
    is_bot = [player0_is_bot, player1_is_bot]
    agents = [Agent() , Agent()]
    
    while not is_terminal(state):
        current_player = state[4]  # Get current player from state tuple
        possible_moves = get_possible_moves(state)
        
        if is_bot[current_player]:
            # Bot player - for now, just choose a random move
            print(get_possible_moves(state))
            bot_move = agents[current_player].select_move(state)
            print(f"Player {current_player} (BOT) chooses move: {bot_move}")
            state = apply_move(state, bot_move)
        else:
            # Human player - get input from terminal
            print(f"Player {current_player}'s turn (HUMAN)")
            print("Available moves:")
            
            # Display moves with indices for selection
            pawn_moves = [m for m in possible_moves if m[0] == "move"]
            fence_moves = [m for m in possible_moves if m[0] == "fence"]
            
            print("\nPawn moves:")
            for i, move in enumerate(pawn_moves):
                print(f"{i+1}. Move to {move[1]}")
            
            print("\nFence moves:")
            # Only show some fence moves if there are many
            if len(fence_moves) > 10:
                for i, move in enumerate(fence_moves[:10]):
                    print(f"{i+len(pawn_moves)+1}. Place {move[2]} fence at {move[1]}")
                print(f"... and {len(fence_moves)-10} more fence options")
            else:
                for i, move in enumerate(fence_moves):
                    print(f"{i+len(pawn_moves)+1}. Place {move[2]} fence at {move[1]}")
            
            # Get user input
            valid_move = False
            while not valid_move:
                try:
                    move_type = input("\nEnter 'p' for pawn move or 'f' for fence placement: ").strip().lower()
                    
                    if move_type == 'p':
                        if not pawn_moves:
                            print("No valid pawn moves available!")
                            continue
                            
                        print("Enter coordinates as 'row,col' (e.g., '3,4'): ")
                        coords = input().strip()
                        row, col = map(int, coords.split(','))
                        
                        chosen_move = ("move", (row, col))
                        if chosen_move in pawn_moves:
                            valid_move = True
                        else:
                            print("Invalid move! Try again.")
                    
                    elif move_type == 'f':
                        if not fence_moves:
                            print("No valid fence moves available!")
                            continue
                            
                        print("Enter fence position as 'row,col' (e.g., '3,4'): ")
                        coords = input().strip()
                        row, col = map(int, coords.split(','))
                        
                        print("Enter fence orientation ('h' for horizontal, 'v' for vertical): ")
                        orientation = input().strip().upper()
                        if orientation.lower() not in ['h', 'v']:
                            print("Invalid orientation! Use 'h' or 'v'.")
                            continue
                        
                        chosen_move = ("fence", (row, col), orientation)
                        if chosen_move in fence_moves:
                            valid_move = True
                        else:
                            print("Invalid fence placement! Try again.")
                    
                    else:
                        print("Invalid input! Enter 'p' for pawn or 'f' for fence.")
                
                except (ValueError, IndexError):
                    print("Invalid input format! Try again.")
            
            # Apply the chosen move
            print(f"Player {current_player} chooses: {chosen_move}")
            state = apply_move(state, chosen_move)
        
        # Display the updated game state
        print_game_state(state)
    
    # Game is over, announce winner
    winner = check_winner(state)
    print(f"Game over! Player {winner} wins!")
    return winner

def main():
    """Run the game with configurable player types"""
    print("Welcome to Quoridor!")
    print("Choose player types:")
    
    try:
        p0_type = input("Is Player 0 a bot? (y/n): ").strip().lower()
        p1_type = input("Is Player 1 a bot? (y/n): ").strip().lower()
        
        player0_is_bot = p0_type == 'y'
        player1_is_bot = p1_type == 'y'
        
        winner = game_loop(player0_is_bot, player1_is_bot)
        print(f"Thanks for playing! Player {winner} won the game.")
    
    except KeyboardInterrupt:
        print("\nGame interrupted. Exiting...")
    # except Exception as e:
    #     print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()