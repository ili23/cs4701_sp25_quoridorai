import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import from game_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_engine.game_state_tuple import (
    BOARD_SIZE, 
    PLAYER_ONE, 
    PLAYER_TWO, 
    create_initial_state,
    get_current_player
)

class QuoridorFeatureExtractor:
    """Extract features from a Quoridor game state for neural network input."""
    
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.feature_size = self._calculate_feature_size()
        
    def _calculate_feature_size(self):
        # Player positions (one-hot encoded): 2 * board_size^2
        # Horizontal fences: (board_size-1)^2
        # Vertical fences: (board_size-1)^2
        # Remaining fences for each player: 2
        # Current player: 1
        return (2 * self.board_size**2 + 
                2 * (self.board_size-1)**2 + 
                2 + 
                1)
    
    def extract_features(self, state):
        """Convert a game state to a tensor of features for the neural network."""
        pawns, fences, h_fences, v_fences, current_player = state
        
        # Initialize features array
        features = np.zeros(self.feature_size)
        
        # One-hot encode player 1 position
        p1_row, p1_col = pawns[PLAYER_ONE]
        p1_index = p1_row * self.board_size + p1_col
        features[p1_index] = 1
        
        # One-hot encode player 2 position
        p2_row, p2_col = pawns[PLAYER_TWO]
        p2_index = self.board_size**2 + p2_row * self.board_size + p2_col
        features[p2_index] = 1
        
        # Horizontal fences
        h_fence_offset = 2 * self.board_size**2
        for i in range(self.board_size-1):
            for j in range(self.board_size-1):
                if h_fences[i][j]:
                    h_idx = i * (self.board_size-1) + j
                    features[h_fence_offset + h_idx] = 1
        
        # Vertical fences
        v_fence_offset = h_fence_offset + (self.board_size-1)**2
        for i in range(self.board_size-1):
            for j in range(self.board_size-1):
                if v_fences[i][j]:
                    v_idx = i * (self.board_size-1) + j
                    features[v_fence_offset + v_idx] = 1
        
        # Remaining fences
        fence_count_offset = v_fence_offset + (self.board_size-1)**2
        features[fence_count_offset] = fences[PLAYER_ONE] / 10.0  # Normalize by max fences
        features[fence_count_offset + 1] = fences[PLAYER_TWO] / 10.0
        
        # Current player
        features[-1] = current_player
        
        return torch.FloatTensor(features)
    
    def get_distance_to_goal(self, state):
        """Calculate the distance from each player to their goal."""
        pawns, _, _, _, _ = state
        
        # Player 1 distance to bottom row
        p1_row, _ = pawns[PLAYER_ONE]
        p1_distance = self.board_size - 1 - p1_row
        
        # Player 2 distance to top row
        p2_row, _ = pawns[PLAYER_TWO]
        p2_distance = p2_row
        
        return p1_distance, p2_distance


class QuoridorNN(nn.Module):
    """Neural network for evaluating Quoridor positions."""
    
    def __init__(self, input_size, hidden_size=256):
        super(QuoridorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output is probability of player 1 winning
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid to get probability between 0 and 1
        return x


class QuoridorNNTrainer:
    """Class for training and using the Quoridor neural network model."""
    
    def __init__(self, learning_rate=0.001):
        self.feature_extractor = QuoridorFeatureExtractor()
        self.model = QuoridorNN(self.feature_extractor.feature_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss
        
    def evaluate_position(self, state):
        """Evaluate a position and return win probability for player 1."""
        self.model.eval()
        with torch.no_grad():
            features = self.feature_extractor.extract_features(state)
            win_prob = self.model(features).item()
            
            # If player 2's turn, return 1 - win_prob for perspective adjustment
            if get_current_player(state) == PLAYER_TWO:
                return 1.0 - win_prob
            return win_prob
    
    def train_step(self, state, target_value):
        """Perform one training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        features = self.feature_extractor.extract_features(state)
        output = self.model(features)
        
        target = torch.FloatTensor([target_value])
        loss = self.loss_fn(output, target)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load the model from a file."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Example usage
if __name__ == "__main__":
    # Create a trainer
    trainer = QuoridorNNTrainer()
    
    # Create an initial state
    state = create_initial_state()
    
    # Evaluate the position
    win_probability = trainer.evaluate_position(state)
    print(f"Win probability for player 1: {win_probability:.4f}")
    
    # Example of training (would normally be done with many game states)
    loss = trainer.train_step(state, 0.5)  # Target value of 0.5 for initial position
    print(f"Training loss: {loss:.4f}")
