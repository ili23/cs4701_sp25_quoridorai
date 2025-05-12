#!/usr/bin/env python3
"""
Generate one-move-to-win training data for Quoridor AI.

This script creates game states where a player is one pawn move away from winning,
assigns them a high evaluation score, and saves them to a CSV file.
"""

import os
import random
import csv
from datetime import datetime

# Add the parent directory to the path so we can import from game_engine
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game_engine.game_state_tuple import (
    BOARD_SIZE,
    PLAYER_ONE,
    PLAYER_TWO,
    create_initial_state,
    apply_move,
    is_terminal,
    check_winner
)

from model.data_loader import QuoridorDataset

def save_to_csv(states, targets, file_path=None):
    """
    Save game states and targets to a CSV file.
    
    Args:
        states: List of game state tuples
        targets: List of target evaluation values
        file_path: Path to output CSV file
    
    Returns:
        Path to the saved CSV file
    """
    if not file_path:
        # Generate a filename with timestamp if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"training_data/dataset_{timestamp}.csv"
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row
        header = [
            'player0_pawn',
            'player1_pawn',
            'num_walls_player0', 'num_walls_player1',
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
        
        writer.writerow(header)
        
        # Write data rows
        for i in range(len(states)):
            state = states[i]
            target = targets[i]
            
            # Extract components from state tuple
            pawns, fences, h_fences, v_fences, current_player, move_count = state
            
            # Format pawn positions using pipe separator
            player0_pawn = f"{pawns[0][0]}|{pawns[0][1]}"
            player1_pawn = f"{pawns[1][0]}|{pawns[1][1]}"
            
            # Create row for CSV with basic features
            row = [
                player0_pawn,            # Player 0 pawn position as "row|col"
                player1_pawn,            # Player 1 pawn position as "row|col"
                fences[0], fences[1],    # Wall counts for player 0 and player 1
                move_count, current_player # Move count and current player
            ]
            
            # Format horizontal walls - 8 columns, each with 8 values separated by |
            for col in range(8):
                h_wall_col = []
                for row_idx in range(8):
                    # Get wall value (0 or 1) for this position
                    wall_value = 1 if h_fences[row_idx][col] else 0
                    h_wall_col.append(str(wall_value))
                # Join values with | and add to row
                row.append('|'.join(h_wall_col))
            
            # Format vertical walls - 8 columns, each with 8 values separated by |
            for col in range(8):
                v_wall_col = []
                for row_idx in range(8):
                    # Get wall value (0 or 1) for this position
                    wall_value = 1 if v_fences[row_idx][col] else 0
                    v_wall_col.append(str(wall_value))
                # Join values with | and add to row
                row.append('|'.join(v_wall_col))
            
            # Add outcome (target value already in -1 to 1 range)
            row.append(target)
            
            writer.writerow(row)
    
    print(f"Dataset saved to {file_path}")
    return file_path

def create_one_move_to_win_state(player=PLAYER_ONE, board_size=BOARD_SIZE):
    """
    Create a game state where the specified player can win with one more move.
    
    Args:
        player: The player who is about to win (PLAYER_ONE/0 or PLAYER_TWO/1)
        board_size: Size of the Quoridor board
    
    Returns:
        A game state tuple where the player can win with one move
    """
    # Get the goal row for the player
    goal_row = board_size - 1 if player == PLAYER_ONE else 0
    
    # Set player's position one row away from goal
    player_row = goal_row - 1 if player == PLAYER_ONE else goal_row + 1
    player_col = random.randint(0, board_size - 1)
    
    # Set opponent's position (far away from goal)
    opponent = PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE
    
    # Ensure opponent is not on their goal row (which would end the game)
    opponent_goal_row = 0 if opponent == PLAYER_ONE else board_size - 1
    
    # Pick a row for opponent that's not their goal row
    if opponent == PLAYER_ONE:
        # If opponent is player 0, keep them away from row board_size-1 (bottom)
        opponent_row = random.randint(0, board_size - 2)
    else:
        # If opponent is player 1, keep them away from row 0 (top)
        opponent_row = random.randint(1, board_size - 1)
    
    opponent_col = random.randint(0, board_size - 1)
    
    # Create the pawns tuple with indexed positions [player0_pos, player1_pos]
    pawns = [None, None]
    pawns[PLAYER_ONE] = (player_row, player_col) if player == PLAYER_ONE else (opponent_row, opponent_col)
    pawns[PLAYER_TWO] = (opponent_row, opponent_col) if player == PLAYER_ONE else (player_row, player_col)
    pawns = tuple(pawns)
    
    # Set fence counts to 10 for both players (full complement) since no walls are placed
    # fences[0] is for player 0, fences[1] is for player 1
    fences = [10, 10]
    
    # Create empty wall states (8x8 grid for standard 9x9 board)
    h_fences = tuple(tuple(False for _ in range(board_size - 1)) for _ in range(board_size - 1))
    v_fences = tuple(tuple(False for _ in range(board_size - 1)) for _ in range(board_size - 1))
    
    # Keeping the board completely empty of walls to create clean one-move-to-win scenarios
    
    # Set current player (0 or 1)
    current_player = player
    
    # Set move count (random but reasonable)
    move_count = random.randint(20, 60)
    
    # Create the state tuple
    state = (pawns, fences, h_fences, v_fences, current_player, move_count)
    
    return state

def generate_one_move_to_win_dataset(num_states=100):
    """
    Generate a dataset of one-move-to-win states.
    
    Args:
        num_states: Number of states to generate
    
    Returns:
        List of tuples (state, target) for training
    """
    training_data = []
    
    for _ in range(num_states):
        # Randomly choose which player is about to win (player 0 or player 1)
        player = random.choice([PLAYER_ONE, PLAYER_TWO])
        
        # Create a state where the player can win with one move
        state = create_one_move_to_win_state(player)
        
        # Target is 1.0 (win) since this is a winning position
        target = 1.0
        
        training_data.append((state, target))
    
    return training_data

def main():
    """Generate one-move-to-win training data and save to CSV."""
    print("Generating one-move-to-win training data for players 0 and 1...")
    
    # Generate training data
    training_data = generate_one_move_to_win_dataset(num_states=200)
    
    # Extract states and targets
    states = [s for s, _ in training_data]
    targets = [t for _, t in training_data]
    
    # Create directory for training data
    save_dir = os.path.join(os.path.dirname(__file__), "model", "training_data")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(save_dir, "fake_training_data.csv")
    save_to_csv(states, targets, output_path)
    
    print(f"Generated {len(training_data)} one-move-to-win states")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
