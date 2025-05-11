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
from game_engine.game_state_tuple import BOARD_SIZE

# Import the dataset class for feature extraction
from model.data_loader import QuoridorDataset


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
        
        # Fully connected layer for move count
        self.fc_move_count = nn.Linear(1, 16)
        
        # Combined fully connected layers
        self.fc_combined = nn.Linear(256 + 32 + 16, 128)  # Updated to include move count features
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
        # Unpack input: x is a tuple (board_tensor, fence_counts, move_count)
        if len(x) == 3:
            board_tensor, fence_counts, move_count = x
        else:
            # Backward compatibility: handle case where move_count is not provided
            board_tensor, fence_counts = x
            # Create a dummy move count tensor (zeros)
            move_count = torch.zeros_like(fence_counts[:, :1])
        
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
        
        # Process move count feature
        x_move_count = F.relu(self.fc_move_count(move_count))
        
        # Combine features
        x_combined = torch.cat((x_board, x_fence, x_move_count), dim=1)
        x_combined = F.relu(self.bn_fc3(self.fc_combined(x_combined)))
        x_combined = self.dropout(x_combined)
        
        # Output layer
        output = torch.tanh(self.fc_output(x_combined))  # Tanh to get value between -1 and 1
        
        return output


class QuoridorLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training Quoridor neural network."""
    
    def __init__(self, learning_rate=0.001):
        super(QuoridorLightningModule, self).__init__()
        self.dataset = QuoridorDataset()  # Only used for feature extraction
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
            # Get features for CNN using the dataset's extract_features_2d method
            board_tensor, fence_counts, move_count_tensor = self.dataset.extract_features_2d(
                state, standardize_perspective=True
            )
            # Add batch dimension for inference
            board_tensor = board_tensor.unsqueeze(0)
            fence_counts = fence_counts.unsqueeze(0)
            move_count_tensor = move_count_tensor.unsqueeze(0)
            features = (board_tensor, fence_counts, move_count_tensor)
                
            eval_score = self(features).item()
            return eval_score 