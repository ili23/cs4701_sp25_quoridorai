import os
import random
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from datetime import datetime

# Add the parent directory to the path so we can import modules properly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import game engine modules after path is configured
from game_engine.game_state_tuple import (
    create_initial_state,
    get_possible_moves,
    apply_move,
    is_terminal,
    check_winner,
    get_current_player
)

# Local imports - use relative imports
from model.cnn import QuoridorLightningModule
from model.data_loader import QuoridorDataModule

# Set default model and checkpoint paths
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "quoridor_cnn_final.ckpt")

class QuoridorTrainer:
    """
    Trainer class for Quoridor CNN using PyTorch Lightning.
    
    This class encapsulates the training process including:
    - Setting up the model
    - Loading training data
    - Configuring callbacks like checkpointing and early stopping
    - Running training & validation
    """
    
    def __init__(self, data_path = None, batch_size = 32, max_epochs = 100):
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        # Create model & data module
        self.model = None
        self.data_module = None
        
        # Make sure checkpoint directory exists
        os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    
    def setup(self):
        """Set up the model and data module for training."""
        # Create model
        self.model = QuoridorLightningModule(learning_rate=0.001)
        
        # Create data module
        self.data_module = QuoridorDataModule(
            data_path=self.data_path,
            batch_size=self.batch_size,
        )
    
    def train(self):
        """Train the model and save the final checkpoint."""
        # First, make sure setup has been called
        if self.model is None or self.data_module is None:
            self.setup()
        
        # Create callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=DEFAULT_MODEL_DIR,
            filename='quoridor-cnn-epoch{epoch:02d}-val_loss{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            default_root_dir=DEFAULT_MODEL_DIR
        )
        
        # Train the model
        trainer.fit(self.model, self.data_module)
        
        # Save the final model
        trainer.save_checkpoint(DEFAULT_MODEL_PATH)
        
        print(f"Training complete! Final model saved to {DEFAULT_MODEL_PATH}")
        
        return {
            "best_model_path": checkpoint_callback.best_model_path,
            "final_model_path": DEFAULT_MODEL_PATH
        }

if __name__ == "__main__":
    print("Training model...")
    trainer = QuoridorTrainer(
        data_path="./model/training_data/gamestate0.csv",
        batch_size=32,
        max_epochs=5
    )
    
    results = trainer.train()
