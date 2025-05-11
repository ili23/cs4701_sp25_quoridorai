import numpy as np
import random
from queue import Queue
from collections import deque
import pickle
import os
import csv
from datetime import datetime

from utility import argmax
from game_state_tuple import get_possible_moves, apply_move, is_terminal, check_winner, get_current_player, BOARD_SIZE, is_fence_between, print_game_state
POSSIBLE_MOVES = {}
SHORTEST_PATHS = {}
def access_cache_or_update(state):
    if state not in POSSIBLE_MOVES:
        POSSIBLE_MOVES[state] = get_possible_moves(state)
    return POSSIBLE_MOVES[state]

def shortest_path(state, player):
    """
    Compute the shortest path from the player's position to their goal using BFS.
    Returns the length of the shortest path.
    
    Uses caching to avoid recomputing paths for the same state-player combination.
    """
    # Check cache first
    cache_key = (state, player)
    if cache_key in SHORTEST_PATHS:
        return SHORTEST_PATHS[cache_key]
    
    pawns, _, h_fences, v_fences, _, _ = state
    
    # Goal row depends on which player we are
    goal_row = BOARD_SIZE - 1 if player == 0 else 0
    
    queue = deque([(pawns[player], 0)])  # (position, distance)
    visited = set([pawns[player]])
    
    while queue:
        (x, y), distance = queue.popleft()
        
        # Check if reached goal row
        if x == goal_row:
            # Store result in cache before returning
            SHORTEST_PATHS[cache_key] = distance
            return distance
            
        # Try all four directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and (nx, ny) not in visited:
                # Check if move is valid (no fence blocking)
                if not is_fence_between(state, (x, y), (nx, ny)):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), distance + 1))
                    
    # If no path is found, return a large value
    result = float('inf')
    SHORTEST_PATHS[cache_key] = result
    return result

def calculate_pawn_probability(player_path_length, opponent_path_length):
    """
    Calculate the probability of making a pawn move using a sigmoid function:
    P(pawn) = 1/(1+e^(-x/7)) where x is the difference between opponent and player path lengths.
    
    Positive x (opponent path > player path) → higher probability of pawn moves
    Negative x (player path > opponent path) → lower probability of pawn moves
    """
    # Calculate path difference (positive when opponent's path is longer)
    path_difference = opponent_path_length - player_path_length
    
    # Apply sigmoid function: 1/(1+e^(-x/7))
    # The divisor 7 controls how gradually the probability changes with path difference
    sigmoid = 1 / (1 + np.exp(-path_difference / 7))
    return np.clip(sigmoid, 0.2, 0.9)

def rollout(state, max_steps=70):
    t = 0
    while not is_terminal(state) and t < max_steps:
        moves = access_cache_or_update(state)
        current_player = get_current_player(state)
        
        # First, check if we can win immediately
        pawn_moves = [m for m in moves if m[0] == "move"]
        winning_move = find_winning_move(state, pawn_moves)
        if winning_move:
            # If we can win, always take that move
            a = winning_move
        else:
            # Normal move selection...
            # Get pawn moves and fence moves
            fence_moves = [m for m in moves if m[0] == "fence"] # check if there are any fences to place
            
            # If no moves available, return None
            if not moves:
                return None
            
            # Calculate shortest path for both players
            player_path_length = shortest_path(state, current_player)
            opponent_path_length = shortest_path(state, 1 - current_player)
            
            # Calculate the dynamic probability of making a pawn move
            pawn_prob = calculate_pawn_probability(player_path_length, opponent_path_length)
            
            # Choose move type based on calculated probability
            if pawn_moves and random.random() < pawn_prob:
                # With 75% probability, use distance-to-goal heuristic
                if random.random() < 0.75:
                    goal_row = BOARD_SIZE - 1 if current_player == 0 else 0
                    pawn_moves.sort(key=lambda m: abs(m[1][0] - goal_row))
                    a = pawn_moves[0]  # Take move closest to goal
                else:
                    a = random.choice(pawn_moves)  # Random move
            elif fence_moves: 
                a = random.choice(fence_moves)
            else:
                a = random.choice(pawn_moves)
        
        # Apply the chosen move
        state = apply_move(state, a)
        t += 1
    # print("rollout took", t, "steps")
    return check_winner(state)

def find_winning_move(state, moves):
    """
    Checks if any of the given moves leads to an immediate win.
    Returns the winning move if found, otherwise None.
    """
    current_player = get_current_player(state)
    goal_row = BOARD_SIZE - 1 if current_player == 0 else 0
    
    for move in moves:
        if move[0] == "move":  # Only pawn moves can lead to immediate win
            move_pos = move[1]
            if move_pos[0] == goal_row:  # This move reaches the goal row
                return move
                
    return None

class GameTree:
    def __init__(self, start_state, parent=None, state_cache=None):
        self.state = start_state
        self.children = None
        self.parent = parent
        
        # Fix for mutable default argument
        if state_cache is None:
            self.state_cache = {}
        else:
            self.state_cache = state_cache
            
        if self.state not in self.state_cache:
            current_player = get_current_player(start_state)
            player_path = shortest_path(start_state, current_player)
            opponent_path = shortest_path(start_state, 1 - current_player)
            
            # Calculate path difference directly
            path_difference = opponent_path - player_path
            
            self.state_cache[self.state] = {
                "n": 0,
                "w": 0,
                "path_diff": path_difference
            }
        
    def value(self):
        """
        Calculate the UCB1 value of this node, adjusted with the path difference
        using the sigmoid function 1/(1+e^(-x/7)).
        """
        if self.state_cache[self.state]["n"] == 0:
            return np.inf
        
        # Standard UCB1 calculation
        c = np.sqrt(2)  # Exploration parameter
        ucb1 = (self.state_cache[self.state]["w"] / self.state_cache[self.state]["n"]) + \
               c * np.sqrt(np.log(self.state_cache[self.parent.state]["n"]) / self.state_cache[self.state]["n"])
        
        return ucb1 
    
    def expand(self):
        if is_terminal(self.state):
            self.backprop(check_winner(self.state))
        else:
            if self.children is None:
                self.backprop(rollout(self.state))
                
                # Create child nodes for all possible moves
                moves = access_cache_or_update(self.state)
                self.children = []
                
                # First check for any winning moves
                winning_move_index = -1
                current_player = get_current_player(self.state)
                
                for i, move in enumerate(moves):
                    if move[0] == "move":
                        child_state = apply_move(self.state, move)
                        child_node = GameTree(child_state, parent=self, state_cache=self.state_cache)
                        self.children.append(child_node)
                        
                        # Check if this is a winning move
                        if check_winner(child_state) == current_player:
                            winning_move_index = i
                            break
                
                # If we found a winning move, only keep that child
                if winning_move_index >= 0:
                    self.children = [self.children[winning_move_index]]
                else:
                    # Complete adding all remaining children

                    # compute the sigmoid
                    # either pick from pawn moves or fence moves based on the sigmoid
                    pawn_moves = [m for m in moves if m[0] == "move"]
                    fence_moves = [m for m in moves if m[0] == "fence"]
                    opponent_path = shortest_path(self.state, 1 - current_player)
                    player_path = shortest_path(self.state, current_player)
                    sigmoid = np.clip(1 / (1 + np.exp(-(opponent_path - player_path) / 4)), 0.2, 0.9)
                    if random.random() < sigmoid:
                        for move in pawn_moves:
                            child_state = apply_move(self.state, move)
                            self.children.append(GameTree(child_state, parent=self, state_cache=self.state_cache))
                    else:
                        for move in fence_moves:
                            child_state = apply_move(self.state, move)
                            self.children.append(GameTree(child_state, parent=self, state_cache=self.state_cache))
            else:
                # Select child with highest value to expand next
                values = [child.value() for child in self.children]
                self.children[np.argmax(values)].expand()
    
    def backprop(self, winner):
        self.state_cache[self.state]["n"] += 1
        if winner is not None:  # There is a winner or tie
            if winner == 0.5:  # Tie case
                # Add 0.5 points for a tie
                self.state_cache[self.state]["w"] += 0.5
            else:  # Winner case
                # The previous player is the opposite of current player in this state
                previous_player = 1 - get_current_player(self.state)
                # If the winner is the previous player, increment wins
                if previous_player == winner:
                    self.state_cache[self.state]["w"] += 1
        
        if self.parent is not None:
            self.parent.backprop(winner)

class Agent:
    def __init__(self, iters=100):
        self.search_depth = iters
        self.tree = None
    
    def select_move(self, state):
        assert not is_terminal(state)
        
        # First, check if we can win in one move
        possible_moves = access_cache_or_update(state)
        pawn_moves = [m for m in possible_moves if m[0] == "move"]
        winning_move = find_winning_move(state, pawn_moves)
        if winning_move:
            return winning_move
        
        # If no immediate win, use MCTS
        if self.tree is None: 
            self.tree = GameTree(state)
        elif self.tree.state != state:
            self.tree = GameTree(state, state_cache=self.tree.state_cache)

        for _ in range(self.search_depth):
            self.tree.expand()
        
        # Choose move based on a combination of win ratio and path difference
        cache = self.tree.state_cache
        
        return possible_moves[argmax(self.tree.children, key=lambda tree: 0 if cache[tree.state]["n"] == 0 else cache[tree.state]["w"] / cache[tree.state]["n"])]               
    
def generate_self_play_data(num_games=100, search_depth=50, save_dir="training_data"):
    """
    Generate training data from self-play games in CSV format.
    
    Args:
        num_games: Number of self-play games to generate
        search_depth: MCTS iterations per move
        save_dir: Directory to save training data
    
    Returns:
        Path to the saved training data file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize data structures to store training examples
    training_data = []
    
    # Create MCTS agent
    agent = Agent(iters=search_depth)
    
    print(f"Generating {num_games} self-play games...")
    
    for game_idx in range(num_games):
        if game_idx % 10 == 0:
            print(f"Game {game_idx}/{num_games}")
        
        # Initialize game state
        from game_state_tuple import get_initial_state
        state = get_initial_state()
        
        # List to store game history
        game_states = []
        
        # Play game until terminal state
        while not is_terminal(state):
            # Store current state
            print_game_state(state)
            game_states.append(state)
            
            # Select move using MCTS
            selected_move = agent.select_move(state)
            
            # Apply move
            state = apply_move(state, selected_move)
        
        # Get game winner
        winner = check_winner(state)
        
        # Create training examples
        for game_state in game_states:
            current_player = get_current_player(game_state)
            
            # Label: 1 if current player won, 0 if lost, 0.5 if tie
            if winner == 0.5:  # Tie
                outcome = 0.0  # Draw is 0 in the -1 to 1 scale
            else:  # Win or loss
                outcome = 1.0 if winner == current_player else -1.0  # Win is 1, loss is -1
            
            # Extract data for CSV format
            pawns, fences, h_fences, v_fences, current_player, move_count = game_state
            player1_pawn_pos = f"{pawns[0][0]}|{pawns[0][1]}"  # Format as "row|col"
            player2_pawn_pos = f"{pawns[1][0]}|{pawns[1][1]}"  # Format as "row|col"
            num_walls_p1 = fences[0]     # Remaining walls for player 1
            num_walls_p2 = fences[1]     # Remaining walls for player 2
            
            # Create a row for the CSV with basic features
            csv_row = [
                player1_pawn_pos,      # Player 1 pawn position as "row|col"
                player2_pawn_pos,      # Player 2 pawn position as "row|col"
                num_walls_p1,          # Number of walls player 1
                num_walls_p2,          # Number of walls player 2
                move_count,            # Number of moves
                current_player,        # Current player to move
            ]
            
            # Format horizontal walls - 8 columns, each with 8 values separated by |
            for col in range(8):
                h_wall_col = []
                for row in range(8):
                    # Get wall value (0 or 1) for this position
                    wall_value = 1 if h_fences[row][col] else 0
                    h_wall_col.append(str(wall_value))
                # Join values with | and add to row
                csv_row.append('|'.join(h_wall_col))
            
            # Format vertical walls - 8 columns, each with 8 values separated by |
            for col in range(8):
                v_wall_col = []
                for row in range(8):
                    # Get wall value (0 or 1) for this position
                    wall_value = 1 if v_fences[row][col] else 0
                    v_wall_col.append(str(wall_value))
                # Join values with | and add to row
                csv_row.append('|'.join(v_wall_col))
            
            # Add outcome as the target value
            csv_row.append(outcome)
            
            training_data.append(csv_row)
    
    # Save training data to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"self_play_data_{timestamp}.csv")
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row
        header = [
            'player1_pawn',
            'player2_pawn',
            'num_walls_player1', 'num_walls_player2',
            'move_count', 'current_player'
        ]
        
        # Add horizontal wall headers
        for i in range(8):
            header.append(f'h_wall_col{i}')
        
        # Add vertical wall headers
        for i in range(8):
            header.append(f'v_wall_col{i}')
        
        # Add outcome header
        header.append('outcome')
        
        # Write header and data
        writer.writerow(header)
        writer.writerows(training_data)
    
    print(f"Training data saved to {save_path}")
    print(f"Total training examples: {len(training_data)}")
    
    return save_path

if __name__ == "__main__":
    generate_self_play_data(num_games=1, search_depth=10, save_dir="training_data")