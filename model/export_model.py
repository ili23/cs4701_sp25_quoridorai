#!/usr/bin/env python3
"""
Export trained PyTorch model to a format that can be loaded in C++.
"""

import os
import sys
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.cnn import QuoridorLightningModule

def export_model(checkpoint_path="model/checkpoints/quoridor_cnn_final.ckpt", output_path=None):
    """Export a trained model to TorchScript format for C++ loading."""
    
    # Load the model from checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # lightning_model = QuoridorLightningModule.load_from_checkpoint(checkpoint_path)
    lightning_model = QuoridorLightningModule()
    lightning_model.eval()  # Set to evaluation mode
    
    # Get the underlying PyTorch model
    # In QuoridorLightningModule, the actual CNN is stored in self.model
    torch_model = lightning_model.model
    torch_model.eval()
    
    # Create a wrapper function to match the expected input format in C++
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_tuple):
            # Unpack the input tuple (this matches what we're doing in C++)
            board_tensor, fence_counts, move_count = input_tuple
            
            # Forward pass through the model
            # In QuoridorCNN, the forward method already handles the tuple unpacking
            return self.model((board_tensor, fence_counts, move_count))
    
    # Create the wrapper
    wrapped_model = ModelWrapper(torch_model)
    
    # Create example inputs for tracing
    board_tensor = torch.zeros(1, 4, 17, 17)  # Batch size 1, 4 channels, 17x17 grid
    fence_counts = torch.zeros(1, 2)  # Batch size 1, 2 values (current/opponent)
    move_count = torch.zeros(1, 1)  # Batch size 1, 1 value
    
    # Package inputs as a tuple to match our C++ interface
    example_input = (board_tensor, fence_counts, move_count)
    
    # Trace the model using the wrapper
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapped_model, (example_input,))
    
    # Save the traced model
    if output_path is None:
        output_path = os.path.splitext(checkpoint_path)[0] + "_traced.pt"
    
    print(f"Saving traced model to: {output_path}")
    traced_model.save(output_path)
    
    print("Done! The model can now be loaded in C++ using torch::jit::load()")
    return output_path

if __name__ == "__main__":
    exported_path = export_model()
