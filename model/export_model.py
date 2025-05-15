import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.cnn import QuoridorLightningModule



def export_model(checkpoint_path="model/checkpoints/quoridor_cnn_final.ckpt", output_path="model/checkpoints/quoridor_cnn_final_traced.pt"):
    """Export a trained model to TorchScript format for C++ loading."""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    lightning_model = QuoridorLightningModule()
    lightning_model.eval()  # Set to evaluation mode
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
    
    wrapped_model = ModelWrapper(torch_model)
    board_tensor = torch.zeros(1, 4, 9, 9)
    fence_counts = torch.zeros(1, 2)
    move_count = torch.zeros(1, 1)
    example_input = (board_tensor, fence_counts, move_count)
    traced_model = torch.jit.trace(wrapped_model, (example_input,))
    traced_model.save(output_path)
    print("Done! The model can now be loaded in C++ using torch::jit::load()")

if __name__ == "__main__":
    export_model()
