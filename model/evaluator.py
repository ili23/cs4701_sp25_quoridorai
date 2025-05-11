import os
import random
import csv
from tqdm import tqdm
import sys

# Add the parent directory to the path so we can import modules properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import modules after path is configured
from game_engine.game_state_tuple import (
    create_initial_state,
    get_possible_moves,
    apply_move,
    is_terminal,
    check_winner,
    get_current_player
)

# Local import - use relative import
from .cnn import QuoridorLightningModule


class QuoridorEvaluator:
    """Class for evaluating the trained Quoridor neural network."""
    
    def __init__(self, model_path = None):
        if model_path is None:
            # Look for the final model in the checkpoints directory
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
            final_model_path = os.path.join(checkpoint_dir, "quoridor_cnn_final.ckpt")
            
            if os.path.exists(final_model_path):
                model_path = final_model_path
        
        # If a specific model path is provided or found, load it
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = QuoridorLightningModule.load_from_checkpoint(model_path)
        else:
            # Create a new model if no existing model was found
            print("No existing model found. Creating a new model.")
            self.model = QuoridorLightningModule()
    
    def evaluate_against_random(self, num_games=10):
        """Evaluate the model against a random player."""
        wins = 0
        losses = 0
        draws = 0
        
        print(f"Evaluating model over {num_games} games...")
        
        for game in tqdm(range(num_games)):
            state = create_initial_state()
            move_count = 0
            
            while not is_terminal(state) and move_count < 200:
                current_player = get_current_player(state)
                moves = get_possible_moves(state)
                
                if current_player == 0:  # Neural network player
                    # Use neural network to select move
                    best_score = -float('inf')
                    best_move = None
                    
                    for move in moves:
                        # Apply move
                        next_state = apply_move(state, move)
                        # Evaluate position (board is standardized inside evaluate_position)
                        score = self.model.evaluate_position(next_state)
                        
                        # Track best move
                        if score > best_score:
                            best_score = score
                            best_move = move
                    
                    selected_move = best_move
                else:  # Random player
                    selected_move = random.choice(moves)
                
                # Apply the selected move
                state = apply_move(state, selected_move)
                move_count += 1
            
            # Determine the outcome
            winner = check_winner(state)
            if winner == 0:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1
        
        # Calculate win rate
        win_rate = wins / num_games
        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
        print(f"Win rate: {win_rate:.2f}")
        
        return win_rate
    
    def save_evaluation_results(self, results, filename=None):
        """Save evaluation results to a CSV file."""
        if filename is None:
            filename = "evaluation_results_cnn.csv"
            
        output_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        # Write results to CSV file
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header and data rows
            writer.writerow(['metric', 'value'])
            for key, value in results.items():
                writer.writerow([key, value])
        
        print(f"Evaluation results saved to {output_path}")
    
    def compare_with_baseline(self, baseline_model_path=None, num_games=50):
        """
        Compare the current model with a baseline model.
        
        Args:
            baseline_model_path: Path to the baseline model
            num_games: Number of games to play for comparison
            
        Returns:
            Dictionary with comparison results
        """
        # Create a baseline evaluator
        baseline_evaluator = QuoridorEvaluator(model_path=baseline_model_path)
        
        # Results tracking
        current_model_wins = 0
        baseline_wins = 0
        draws = 0
        
        print(f"Comparing models over {num_games} games...")
        
        for game in tqdm(range(num_games)):
            state = create_initial_state()
            move_count = 0
            
            while not is_terminal(state) and move_count < 200:
                current_player = get_current_player(state)
                moves = get_possible_moves(state)
                
                # Determine which model to use based on player
                if current_player == 0:  # Current model plays as player 0
                    model = self.model
                else:  # Baseline model plays as player 1
                    model = baseline_evaluator.model
                
                # Find best move using the appropriate model
                best_score = -float('inf')
                best_move = None
                
                for move in moves:
                    next_state = apply_move(state, move)
                    score = model.evaluate_position(next_state)
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                
                # Apply the selected move
                state = apply_move(state, best_move)
                move_count += 1
            
            # Determine the outcome
            winner = check_winner(state)
            if winner == 0:
                current_model_wins += 1
            elif winner == 1:
                baseline_wins += 1
            else:
                draws += 1
        
        # Calculate win rates
        current_win_rate = current_model_wins / num_games
        baseline_win_rate = baseline_wins / num_games
        draw_rate = draws / num_games
        
        results = {
            "current_model_wins": current_model_wins,
            "baseline_wins": baseline_wins,
            "draws": draws,
            "current_win_rate": current_win_rate,
            "baseline_win_rate": baseline_win_rate,
            "draw_rate": draw_rate,
            "games_played": num_games
        }
        
        print(f"Current model wins: {current_model_wins}, Baseline wins: {baseline_wins}, Draws: {draws}")
        print(f"Current model win rate: {current_win_rate:.2f}, Baseline win rate: {baseline_win_rate:.2f}")
        
        return results 