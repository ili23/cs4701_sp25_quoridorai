import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import csv
from torch.utils.data import Dataset
import lightning as pl
from typing import List, Tuple
from datetime import datetime

# Add the parent directory to the path so we can import from game_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_engine.game_state_tuple import (
    BOARD_SIZE, 
    PLAYER_ONE, 
    PLAYER_TWO, 
)


class QuoridorFeatureExtractor:
    """Extract features from a Quoridor game state for neural network input."""
    
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.grid_size = 2 * self.board_size - 1  # 17x17 for standard 9x9 board
        
    def extract_features_2d(self, state, standardize_perspective=True):
        """
        Convert a game state to a 2D tensor representation for CNN input using a 17x17 grid.
        
        In the 17x17 grid:
        - Even indices (0, 2, 4, ..., 16) represent cells where pawns can stand
        - Odd indices (1, 3, 5, ..., 15) represent positions where walls can be placed
        
        Args:
            state: The game state tuple
            standardize_perspective: If True, orient the board from bottom player's perspective
        
        Returns:
            Tensor of shape (C, H, W) where C=4 channels:
                - Channel 0: Current player's position (1 where pawn is, 0 elsewhere)
                - Channel 1: Opponent's position (1 where pawn is, 0 elsewhere)
                - Channel 2: Horizontal walls (1 where horizontal walls exist)
                - Channel 3: Vertical walls (1 where vertical walls exist)
            Plus a separate tensor for the fence counts [current_player_fences, opponent_fences]
        """
        pawns, fences, h_fences, v_fences, current_player = state
        
        # Determine if we need to flip the perspective
        flip_perspective = standardize_perspective and current_player == PLAYER_TWO
        
        # Define current and opponent players based on perspective
        if not flip_perspective:
            current_idx, opponent_idx = PLAYER_ONE, PLAYER_TWO
            current_fences, opponent_fences = fences[PLAYER_ONE], fences[PLAYER_TWO]
        else:
            current_idx, opponent_idx = PLAYER_TWO, PLAYER_ONE
            current_fences, opponent_fences = fences[PLAYER_TWO], fences[PLAYER_ONE]
        
        # Initialize the tensor representation with 4 channels on a 17x17 grid
        # Channel 0: Current player's position
        # Channel 1: Opponent's position
        # Channel 2: Horizontal walls
        # Channel 3: Vertical walls
        board_tensor = torch.zeros(4, self.grid_size, self.grid_size)
        
        # Convert pawn positions to 17x17 grid coordinates
        # In 17x17, the pawn positions are at even indices (0, 2, 4, ..., 16)
        current_row, current_col = pawns[current_idx]
        opponent_row, opponent_col = pawns[opponent_idx]
        
        # If flipping perspective, invert the row indices
        if flip_perspective:
            current_row = self.board_size - 1 - current_row
            opponent_row = self.board_size - 1 - opponent_row
        
        # Convert to 17x17 grid coordinates (multiply by 2)
        grid_current_row, grid_current_col = current_row * 2, current_col * 2
        grid_opponent_row, grid_opponent_col = opponent_row * 2, opponent_col * 2
        
        # Set pawn positions in tensor
        board_tensor[0, grid_current_row, grid_current_col] = 1  # Current player
        board_tensor[1, grid_opponent_row, grid_opponent_col] = 1  # Opponent
        
        # Place horizontal walls on the 17x17 grid - Channel 2
        # Horizontal walls occupy odd row, even column indices on the 17x17 grid
        for i in range(self.board_size - 1):
            for j in range(self.board_size - 1):
                if flip_perspective:
                    # Flip row index for horizontal fences if needed
                    flipped_i = (self.board_size - 2) - i
                    has_wall = h_fences[flipped_i][j]
                else:
                    has_wall = h_fences[i][j]
                
                if has_wall:
                    # Convert to 17x17 grid coordinates
                    # Horizontal walls are at (2i+1, 2j)
                    grid_row, grid_col = 2*i + 1, 2*j
                    # Set wall position in tensor - channel 2 for horizontal walls
                    board_tensor[2, grid_row, grid_col] = 1
                    board_tensor[2, grid_row, grid_col + 1] = 1
                    board_tensor[2, grid_row, grid_col + 2] = 1
        
        # Place vertical walls on the 17x17 grid - Channel 3
        # Vertical walls occupy even row, odd column indices on the 17x17 grid
        for i in range(self.board_size - 1):
            for j in range(self.board_size - 1):
                if flip_perspective:
                    # Flip row index for vertical fences if needed
                    flipped_i = (self.board_size - 2) - i
                    has_wall = v_fences[flipped_i][j]
                else:
                    has_wall = v_fences[i][j]
                
                if has_wall:
                    # Convert to 17x17 grid coordinates
                    # Vertical walls are at (2i, 2j+1)
                    grid_row, grid_col = 2*i, 2*j + 1
                    # Set wall position in tensor - channel 3 for vertical walls
                    board_tensor[3, grid_row, grid_col] = 1
                    board_tensor[3, grid_row + 1, grid_col] = 1
                    board_tensor[3, grid_row + 2, grid_col] = 1
        
        # Create fence count tensor
        fence_counts = torch.FloatTensor([current_fences / 10.0, opponent_fences / 10.0])
        
        return board_tensor, fence_counts
    
    def get_distance_to_goal(self, state, standardize_perspective=True):
        """
        Calculate the distance from each player to their goal.
        
        Args:
            state: The game state tuple
            standardize_perspective: If True, orient from bottom player's perspective
        
        Returns:
            Tuple of (current_player_distance, opponent_distance)
        """
        pawns, _, _, _, current_player = state
        
        # Determine if we need to flip the perspective
        flip_perspective = standardize_perspective and current_player == PLAYER_TWO
        
        if not flip_perspective:
            # Standard orientation
            current_idx, opponent_idx = PLAYER_ONE, PLAYER_TWO
        else:
            # Flipped orientation
            current_idx, opponent_idx = PLAYER_TWO, PLAYER_ONE
        
        # Current player distance to goal
        current_row, _ = pawns[current_idx]
        if flip_perspective:
            # If flipped, the current player is trying to reach row 0
            current_distance = current_row
        else:
            # If not flipped, the current player is trying to reach row (board_size-1)
            current_distance = self.board_size - 1 - current_row
        
        # Opponent distance to goal
        opponent_row, _ = pawns[opponent_idx]
        if flip_perspective:
            # If flipped, the opponent is trying to reach row (board_size-1)
            opponent_distance = self.board_size - 1 - opponent_row
        else:
            # If not flipped, the opponent is trying to reach row 0
            opponent_distance = opponent_row
        
        return current_distance, opponent_distance


class QuoridorDataset(Dataset):
    """Dataset for Quoridor game states and their evaluation targets."""
    
    def __init__(self, data_path: str = None, states: List = None, targets: List = None):
        self.feature_extractor = QuoridorFeatureExtractor()
        
        if data_path and os.path.exists(data_path):
            self.load_from_file(data_path)
        elif states and targets:
            self.states = states
            self.targets = targets
        else:
            self.states = []
            self.targets = []
    
    def load_from_file(self, data_path: str):
        """Load game states and targets from a CSV file."""
        if not data_path.endswith('.csv'):
            raise ValueError(f"Unsupported file format. Expected CSV file, got: {data_path}")
        self.load_from_csv(data_path)
    
    def load_from_csv(self, data_path: str):
        """Load game states and targets from a CSV file."""
        self.states = []
        self.targets = []
        
        with open(data_path, 'r', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read header row
            
            for row in reader:
                # Extract the target value (last column)
                target = float(row[-1])
                # Store target directly since it's already in the -1 to 1 range
                self.targets.append(target)
                
                # Extract game state components
                # Parse player positions (using pipe separator)
                player1_coords = row[0].split('|')
                player2_coords = row[1].split('|')
                player1_pawn = (int(player1_coords[0]), int(player1_coords[1]))
                player2_pawn = (int(player2_coords[0]), int(player2_coords[1]))
                pawns = (player1_pawn, player2_pawn)
                
                # Parse numerical values
                num_walls_p1 = int(row[2])
                num_walls_p2 = int(row[3])
                fences = (num_walls_p1, num_walls_p2)
                
                move_count = int(row[4])
                current_player = int(row[5])
                
                # Extract horizontal walls (8x8 grid)
                h_fences = []
                for row_idx in range(8):
                    h_row = []
                    for col_idx in range(8):
                        # Each horizontal wall column is at index 6+col_idx in the CSV row
                        # Each column contains 8 wall values separated by |
                        # We need the value at position row_idx in this column
                        wall_values = row[6 + col_idx].split('|')
                        h_row.append(bool(int(wall_values[row_idx])))
                    h_fences.append(tuple(h_row))
                h_fences = tuple(h_fences)
                
                # Extract vertical walls (8x8 grid)
                v_fences = []
                for row_idx in range(8):
                    v_row = []
                    for col_idx in range(8):
                        # Each vertical wall column is at index 14+col_idx in the CSV row
                        # Each column contains 8 wall values separated by |
                        # We need the value at position row_idx in this column
                        wall_values = row[14 + col_idx].split('|')
                        v_row.append(bool(int(wall_values[row_idx])))
                    v_fences.append(tuple(v_row))
                v_fences = tuple(v_fences)
                
                # Construct the complete state tuple
                state = (pawns, fences, h_fences, v_fences, current_player, move_count)
                self.states.append(state)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        target = self.targets[idx]
        
        # Get features for CNN
        board_tensor, fence_counts = self.feature_extractor.extract_features_2d(
            state, standardize_perspective=True
        )
        target_tensor = torch.FloatTensor([target])
        
        return (board_tensor, fence_counts), target_tensor
    
    def add_game_data(self, game_data: List[Tuple]):
        """Add data from a played game."""
        for state, target in game_data:
            self.states.append(state)
            self.targets.append(target)
    
    def save_to_csv(self, file_path: str):
        """Save the dataset to a CSV file."""
        if not file_path:
            # Generate a filename with timestamp if not provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"training_data/dataset_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', newline='') as f:
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
            
            writer.writerow(header)
            
            # Write data rows
            for i in range(len(self.states)):
                state = self.states[i]
                target = self.targets[i]
                
                # Extract components from state tuple
                pawns, fences, h_fences, v_fences, current_player, move_count = state
                
                # Format pawn positions using pipe separator
                player1_pawn = f"{pawns[0][0]}|{pawns[0][1]}"
                player2_pawn = f"{pawns[1][0]}|{pawns[1][1]}"
                
                # Create row for CSV with basic features
                row = [
                    player1_pawn,            # Player 1 pawn position as "row|col"
                    player2_pawn,            # Player 2 pawn position as "row|col"
                    fences[0], fences[1],    # Wall counts
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


class QuoridorCNN(nn.Module):
    """
    Convolutional Neural Network for evaluating Quoridor positions.
    
    The architecture processes 2D board representations using convolutional layers
    to capture spatial patterns like wall structures and pawn positioning.
    """
    
    def __init__(self, grid_size=17):
        super(QuoridorCNN, self).__init__()
        
        # Convolutional layers - updated for 4 input channels
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = 64 * grid_size * grid_size
        
        # Fully connected layers for board features
        self.fc_board1 = nn.Linear(conv_output_size, 512)
        self.fc_board2 = nn.Linear(512, 256)
        
        # Fully connected layer for fence counts
        self.fc_fence = nn.Linear(2, 32)
        
        # Combined fully connected layers
        self.fc_combined = nn.Linear(256 + 32, 128)
        self.fc_output = nn.Linear(128, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(128)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Unpack input: x is a tuple (board_tensor, fence_counts)
        board_tensor, fence_counts = x
        
        # Process board with convolutional layers
        x_board = F.relu(self.bn1(self.conv1(board_tensor)))
        x_board = F.relu(self.bn2(self.conv2(x_board)))
        x_board = F.relu(self.bn3(self.conv3(x_board)))
        
        # Flatten for fully connected layers
        x_board = x_board.view(x_board.size(0), -1)
        
        # Process board features
        x_board = F.relu(self.bn_fc1(self.fc_board1(x_board)))
        x_board = self.dropout(x_board)
        x_board = F.relu(self.bn_fc2(self.fc_board2(x_board)))
        x_board = self.dropout(x_board)
        
        # Process fence count features
        x_fence = F.relu(self.fc_fence(fence_counts))
        
        # Combine features
        x_combined = torch.cat((x_board, x_fence), dim=1)
        x_combined = F.relu(self.bn_fc3(self.fc_combined(x_combined)))
        x_combined = self.dropout(x_combined)
        
        # Output layer
        output = torch.tanh(self.fc_output(x_combined))  # Tanh to get value between -1 and 1
        
        return output


class QuoridorLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training Quoridor neural network."""
    
    def __init__(self, learning_rate=0.001):
        super(QuoridorLightningModule, self).__init__()
        self.feature_extractor = QuoridorFeatureExtractor()
        self.learning_rate = learning_rate
        # Initialize model with 17x17 grid size
        self.model = QuoridorCNN(grid_size=2 * BOARD_SIZE - 1)
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        features, targets = batch
        outputs = self(features)
        loss = F.mse_loss(outputs, targets)  # Use MSE loss for regression between -1 and 1
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, targets = batch
        outputs = self(features)
        loss = F.mse_loss(outputs, targets)  # Use MSE loss for regression between -1 and 1
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def evaluate_position(self, state):
        """
        Evaluate a position and return score between -1 and 1.
        
        For this to work correctly, the state should be from the current player's perspective.
        The function will standardize the perspective internally.
        
        Args:
            state: The game state tuple
        
        Returns:
            Score between -1 and 1, where 1 is win for current player, -1 is loss
        """
        self.eval()
        with torch.no_grad():
            # Get features for CNN
            board_tensor, fence_counts = self.feature_extractor.extract_features_2d(
                state, standardize_perspective=True
            )
            # Add batch dimension for inference
            board_tensor = board_tensor.unsqueeze(0)
            fence_counts = fence_counts.unsqueeze(0)
            features = (board_tensor, fence_counts)
                
            eval_score = self(features).item()
            return eval_score 