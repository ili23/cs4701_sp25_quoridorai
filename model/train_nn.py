import sys
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_engine.game_state_tuple import (
    create_initial_state,
    get_possible_moves,
    apply_move,
    is_terminal,
    check_winner,
    get_current_player
)
from model.nn import QuoridorNNTrainer

def self_play_game(nn_trainer, epsilon=0.1, max_moves=200):
    """Play a game against self using the neural network for move selection.
    
    Args:
        nn_trainer: The neural network trainer
        epsilon: Probability of selecting a random move (exploration)
        max_moves: Maximum number of moves before declaring a draw
        
    Returns:
        List of (state, winner) tuples for training
    """
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
                # Evaluate position
                score = nn_trainer.evaluate_position(next_state)
                
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
        # Target value: 1 if current player won, 0 if lost, 0.5 if draw
        if winner is None:
            target = 0.5  # Draw
        elif winner == current_player:
            target = 1.0  # Win
        else:
            target = 0.0  # Loss
        
        training_data.append((game_state, target))
    
    return training_data

def train_network(episodes=100, batch_size=32, epsilon_start=0.5, epsilon_end=0.1):
    """Train the neural network using self-play.
    
    Args:
        episodes: Number of episodes to play
        batch_size: Number of states to use for each training update
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
    """
    nn_trainer = QuoridorNNTrainer()
    training_data = []
    losses = []
    
    print(f"Training for {episodes} episodes...")
    
    for episode in tqdm(range(episodes)):
        # Calculate epsilon for this episode (linear decay)
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / episodes)
        
        # Play a game
        game_data = self_play_game(nn_trainer, epsilon=epsilon)
        training_data.extend(game_data)
        
        # Train on a batch of data
        if len(training_data) >= batch_size:
            batch = random.sample(training_data, batch_size)
            
            # Train on each state in the batch
            batch_loss = 0
            for state, target in batch:
                loss = nn_trainer.train_step(state, target)
                batch_loss += loss
            
            # Average loss for the batch
            avg_loss = batch_loss / batch_size
            losses.append(avg_loss)
            
            # Periodically print and save progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Loss: {avg_loss:.4f}")
                
            # Save model periodically
            if episode % 50 == 0 and episode > 0:
                model_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
                os.makedirs(model_dir, exist_ok=True)
                nn_trainer.save_model(os.path.join(model_dir, f"quoridor_nn_ep{episode}.pt"))
    
    # Save final model
    model_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    nn_trainer.save_model(os.path.join(model_dir, "quoridor_nn_final.pt"))
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(model_dir, 'training_loss.png'))
    plt.close()
    
    return nn_trainer

def evaluate_model(model_path=None, num_games=10):
    """Evaluate the trained model against a random agent.
    
    Args:
        model_path: Path to the trained model (optional)
        num_games: Number of games to play for evaluation
        
    Returns:
        Win rate against random agent
    """
    nn_trainer = QuoridorNNTrainer()
    
    # Load model if provided
    if model_path and os.path.exists(model_path):
        nn_trainer.load_model(model_path)
    
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
                    # Evaluate position
                    score = nn_trainer.evaluate_position(next_state)
                    
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate Quoridor neural network')
    parser.add_argument('--train', action='store_true', help='Train the neural network')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the neural network')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for training')
    parser.add_argument('--model', type=str, help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting training...")
        start_time = time.time()
        nn_trainer = train_network(episodes=args.episodes)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    if args.evaluate:
        print("Starting evaluation...")
        model_path = args.model
        if args.model is None and os.path.exists(os.path.join(os.path.dirname(__file__), "checkpoints", "quoridor_nn_final.pt")):
            model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "quoridor_nn_final.pt")
        
        win_rate = evaluate_model(model_path=model_path)
        print(f"Evaluation complete. Win rate: {win_rate:.2f}")
    
    if not args.train and not args.evaluate:
        print("No action specified. Use --train to train or --evaluate to evaluate.")
        print("Example: python train_nn.py --train --episodes 200")
        print("Example: python train_nn.py --evaluate --model checkpoints/quoridor_nn_final.pt") 