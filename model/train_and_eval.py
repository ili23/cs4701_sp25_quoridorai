import os
import argparse
import matplotlib.pyplot as plt
import time

from model.cnn import QuoridorLightningModule, QuoridorDataset
from model.trainer import QuoridorTrainer
from model.evaluator import QuoridorEvaluator


def visualize_training_progress(loss_history, save_path="training_loss.png"):
    """Visualize training loss over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(save_path)
    plt.close()
    print(f"Training visualization saved to {save_path}")


def main():
    """Main function for training and evaluating the Quoridor neural network."""
    parser = argparse.ArgumentParser(description='Train and evaluate Quoridor neural network with CNN')
    parser.add_argument('--train', action='store_true', help='Train the CNN model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the CNN model')
    parser.add_argument('--self-play', action='store_true', help='Train with self-play')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for self-play training')
    parser.add_argument('--model', type=str, help='Path to model for evaluation')
    parser.add_argument('--data', type=str, help='Path to training data CSV file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--eval-games', type=int, default=100, help='Number of games to play for evaluation')
    parser.add_argument('--compare', action='store_true', help='Compare with a baseline model')
    parser.add_argument('--baseline-model', type=str, help='Path to baseline model for comparison')
    
    args = parser.parse_args()
    
    # Find training data
    data_path = args.data
    if not data_path:
        # Look for CSV files
        training_dir = os.path.join(os.path.dirname(__file__), "training_data")
        csv_files = [f for f in os.listdir(training_dir) if f.endswith('.csv')] if os.path.exists(training_dir) else []
        
        if csv_files:
            # Use the most recent CSV file (assuming filenames contain timestamps)
            data_path = os.path.join(training_dir, sorted(csv_files)[-1])
        else:
            print("No CSV training data files found.")
    
    if args.train:
        if not data_path:
            print("Error: No training data file specified or found. Please provide a CSV file with --data argument.")
            return
        
        print(f"Starting CNN training with data from {data_path}...")
        start_time = time.time()
        trainer = QuoridorTrainer(
            data_path=data_path,
            batch_size=args.batch_size, 
            max_epochs=args.epochs
        )
        model = trainer.train()
        
        training_time = time.time() - start_time
        print(f"CNN training completed in {training_time:.2f} seconds!")
        
        # Automatically run evaluation after training if requested
        if args.evaluate:
            print(f"Running post-training evaluation of CNN model...")
            model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "quoridor_cnn_final.ckpt")
            evaluator = QuoridorEvaluator(model_path=model_path)
            win_rate = evaluator.evaluate_against_random(num_games=args.eval_games)
            
            results = {
                "win_rate": win_rate,
                "training_time": training_time,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            }
            
            evaluator.save_evaluation_results(results)
    
    if args.self_play and not args.train:  # Only run if --train is not also specified
        print(f"Starting CNN training with self-play...")
        start_time = time.time()
        trainer = QuoridorTrainer(
            data_path=data_path,
            batch_size=args.batch_size, 
            max_epochs=args.epochs
        )
        model = trainer.train_with_self_play(episodes=args.episodes)
        
        training_time = time.time() - start_time
        print(f"CNN self-play training completed in {training_time:.2f} seconds!")
        
        # Automatically run evaluation after training if requested
        if args.evaluate:
            print(f"Running post-training evaluation of CNN model...")
            model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "quoridor_cnn_final.ckpt")
            evaluator = QuoridorEvaluator(model_path=model_path)
            win_rate = evaluator.evaluate_against_random(num_games=args.eval_games)
            
            results = {
                "win_rate": win_rate,
                "training_time": training_time,
                "episodes": args.episodes,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            }
            
            evaluator.save_evaluation_results(results)
    
    if args.evaluate and not args.train and not args.self_play:
        print(f"Starting evaluation of CNN model...")
        evaluator = QuoridorEvaluator(model_path=args.model)
        win_rate = evaluator.evaluate_against_random(num_games=args.eval_games)
        
        results = {
            "win_rate": win_rate,
            "model_path": args.model
        }
        
        evaluator.save_evaluation_results(results)
    
    if args.compare:
        print("Comparing CNN models...")
        evaluator = QuoridorEvaluator(model_path=args.model)
        
        # Determine baseline model path if not provided
        baseline_model_path = args.baseline_model
        if not baseline_model_path:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
            potential_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt") and f != "quoridor_cnn_final.ckpt"]
            if potential_files:
                baseline_model_path = os.path.join(checkpoint_dir, potential_files[0])
        
        # Run comparison if we have a baseline model
        if baseline_model_path and os.path.exists(baseline_model_path):
            results = evaluator.compare_with_baseline(
                baseline_model_path=baseline_model_path,
                num_games=args.eval_games
            )
            
            # Save comparison results
            evaluator.save_evaluation_results(
                results, 
                filename=f"comparison_results.csv"
            )
        else:
            print("No baseline model found for comparison. Please train multiple CNN models first.")
    
    if not args.train and not args.self_play and not args.evaluate and not args.compare:
        print("No action specified. Use one of these options:")
        print("  --train              Train the CNN model")
        print("  --self-play          Train with self-play")
        print("  --evaluate           Evaluate the CNN model")
        print("  --compare            Compare with a baseline model")
        print("\nExamples:")
        print("  python -m model.train_and_eval --train --data model/training_data/self_play_data_20250501_180949.csv")
        print("  python -m model.train_and_eval --self-play --episodes 100")
        print("  python -m model.train_and_eval --evaluate --model model/checkpoints/quoridor_cnn_final.ckpt")
        print("  python -m model.train_and_eval --compare --baseline-model model/checkpoints/quoridor_cnn_epoch10.ckpt")


if __name__ == "__main__":
    main() 