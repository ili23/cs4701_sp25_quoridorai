import os
import random
import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

# Add the parent directory to the path so we can import from game_engine
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_engine.game_state_tuple import (
    create_initial_state,
    get_possible_moves,
    apply_move,
    is_terminal,
    check_winner,
    get_current_player
)

from model.cnn import QuoridorDataset, QuoridorLightningModule


class QuoridorTrainer:
    """Class for training Quoridor neural network using PyTorch Lightning."""
    
    def __init__(self, data_path: str = None, batch_size: int = 32, max_epochs: int = 100):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        # Create dataset and dataloader
        self.dataset = QuoridorDataset(data_path=data_path)
        
        # Create CNN model
        self.model = QuoridorLightningModule()
        
        # Setup checkpoint directory
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self):
        """Train the model using PyTorch Lightning."""
        # Split dataset into train and validation
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Define checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename='quoridor_cnn-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback],
            default_root_dir=self.checkpoint_dir
        )
        
        # Train the model
        trainer.fit(self.model, train_loader, val_loader)
        
        # Save final model
        final_model_path = os.path.join(self.checkpoint_dir, "quoridor_cnn_final.ckpt")
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        return self.model
    
    def self_play_game(self, epsilon=0.1, max_moves=200):
        """Play a game against self using the neural network for move selection."""
        state = create_initial_state()
        move_count = 0
        game_states = []
        
        while not is_terminal(state) and move_count < max_moves:
            # Get possible moves
            moves = get_possible_moves(state)
            
            # Epsilon-greedy strategy
            if random.random() < epsilon:
                # Random move
                selected_move = random.choice(moves)
            else:
                # Use neural network to evaluate moves
                best_score = -float('inf')
                best_move = None
                
                for move in moves:
                    # Apply move
                    next_state = apply_move(state, move)
                    # Evaluate position - standardization happens inside the evaluate_position method
                    score = self.model.evaluate_position(next_state)
                    
                    # Track best move
                    if score > best_score:
                        best_score = score
                        best_move = move
                
                selected_move = best_move
            
            # Store the state before applying move
            game_states.append(state)
            
            # Apply the selected move
            state = apply_move(state, selected_move)
            move_count += 1
        
        # Determine winner
        winner = check_winner(state)
        
        # Create training data with winner as target
        training_data = []
        for game_state in game_states:
            current_player = get_current_player(game_state)
            # Target values: 1 for win, -1 for loss, 0 for draw (using -1 to 1 scale)
            if winner is None:
                target = 0.0  # Draw
            elif winner == current_player:
                target = 1.0  # Win
            else:
                target = -1.0  # Loss
            
            training_data.append((game_state, target))
        
        return training_data
    
    def train_with_self_play(self, episodes=100, epsilon_start=0.5, epsilon_end=0.1):
        """Train using both preloaded data and self-play."""
        # Load existing dataset
        if len(self.dataset) > 0:
            # Initial training on existing data
            self.train()
        
        # Self-play training
        for episode in tqdm(range(episodes)):
            # Calculate epsilon for this episode (linear decay)
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / episodes)
            
            # Play a game and get training data
            game_data = self.self_play_game(epsilon=epsilon)
            
            # Add game data to dataset
            self.dataset.add_game_data(game_data)
            
            # Train on updated dataset every 10 episodes
            if (episode + 1) % 10 == 0:
                self.train()
        
        return self.model 