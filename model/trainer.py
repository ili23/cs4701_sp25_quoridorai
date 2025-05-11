import os
import random
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm
from datetime import datetime

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

from model.cnn import QuoridorLightningModule
from model.data_loader import QuoridorDataModule


class QuoridorTrainer:
    """Class for training Quoridor neural network using PyTorch Lightning."""
    
    def __init__(self, data_path: str = None, batch_size: int = 32, max_epochs: int = 100):
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        # Create data module using the QuoridorDataModule
        self.data_module = QuoridorDataModule(
            data_path=data_path,
            batch_size=batch_size,
            val_split=0.2,    # 20% validation split
        )
        
        # Create CNN model
        self.model = QuoridorLightningModule()
        
        # Setup checkpoint directory
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self):
        """Train the model using PyTorch Lightning."""
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
        
        # Train the model using the data module
        trainer.fit(model=self.model, datamodule=self.data_module)
        
        # Save final model
        final_model_path = os.path.join(self.checkpoint_dir, "quoridor_cnn_final.ckpt")
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        return self.model
    
if __name__ == "__main__":
    trainer = QuoridorTrainer(data_path="model/training_data/fake_training_data.csv", batch_size=32, max_epochs=5)
    trainer.train()
