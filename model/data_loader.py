import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import csv
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as pl
from typing import List, Tuple, Optional
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_engine.game_state_tuple import (
    BOARD_SIZE, 
    PLAYER_ONE, 
    PLAYER_TWO, 
)


class QuoridorDataset(Dataset):
    """Dataset for Quoridor game states and their evaluation targets."""
    
    def __init__(self, data_path: str = None, states: List = None, targets: List = None):
        self.board_size = BOARD_SIZE
        self.grid_size = 2 * self.board_size - 1  # 17x17 for standard 9x9 board
        
        if data_path and os.path.exists(data_path):
            if not data_path.endswith('.csv'):
                raise ValueError(f"Unsupported file format. Expected CSV file, got: {data_path}")
            self.load_from_csv(data_path)
        elif states and targets:
            self.states = states
            self.targets = targets
        else:
            self.states = []
            self.targets = []
    
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
            Plus separate tensors for:
                - Fence counts [current_player_fences, opponent_fences]
                - Move count [move_count]
        """
        pawns, fences, h_fences, v_fences, current_player, move_count = state
        
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
        
        # If flipping perspective, invert both row and column indices
        if flip_perspective:
            current_row = self.board_size - 1 - current_row
            opponent_row = self.board_size - 1 - opponent_row
            current_col = self.board_size - 1 - current_col
            opponent_col = self.board_size - 1 - opponent_col
        
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
        
        # Create move count tensor (normalize to 0-1 range)
        # Assuming max game length of 200 moves
        move_count_tensor = torch.FloatTensor([move_count / 200.0])
        
        return board_tensor, fence_counts, move_count_tensor
    
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
        board_tensor, fence_counts, move_count_tensor = self.extract_features_2d(
            state, standardize_perspective=True
        )
        target_tensor = torch.FloatTensor([target])
        
        return (board_tensor, fence_counts, move_count_tensor), target_tensor
      

class QuoridorDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Quoridor dataset.
    Handles loading, splitting, and batch preparation.
    """
    
    def __init__(
        self,
        data_path: str = None,
        states: List = None,
        targets: List = None,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 4,
        seed: int = 42
    ):
        super().__init__()
        self.data_path = data_path
        self.states = states
        self.targets = targets
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        
    
    def setup(self, stage: Optional[str] = None):
        """
        Load and split the data.
        This method is called by every accelerator process.
        """
        # Load the dataset
        full_dataset = QuoridorDataset(
            data_path=self.data_path,
            states=self.states,
            targets=self.targets
        )
        
        # Calculate sizes for splits
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size
        
        # Split dataset into train and validation only
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
    
    def train_dataloader(self):
        """Return the training dataloader with the specified batch size."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Return the validation dataloader with the specified batch size."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    

if __name__ == "__main__":
    # Use absolute path to ensure the file can be found
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "model", "training_data", "fake_training_data.csv")
    
    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"Warning: Data file not found at {data_path}")
        print("Please run generate_fake_training_data.py first to create the training data.")
        sys.exit(1)
    
    print(f"Loading data from: {data_path}")
    
    data_module = QuoridorDataModule(
        data_path=data_path,
        batch_size=8,
        val_split=0.2
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Example usage of the dataloaders
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")
    
    for batch in train_loader:
        features, targets = batch
        board_tensor, fence_counts, move_count = features
        print(f"Batch shape: {board_tensor.shape}, Targets shape: {targets.shape}")
        break