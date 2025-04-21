import sys
import os
import random
import numpy as np
from queue import Queue

# Add the parent directory to the path so we can import from game_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_engine.game_state_tuple import (
    get_possible_moves, 
    apply_move, 
    is_terminal, 
    check_winner, 
    get_current_player
)
from game_engine.utility import argmax
from model.nn import QuoridorNNTrainer

# Cache for possible moves
POSSIBLE_MOVES = {}
def access_cache_or_update(state):
    if state not in POSSIBLE_MOVES:
        POSSIBLE_MOVES[state] = get_possible_moves(state)
    return POSSIBLE_MOVES[state]

class NNGameTree:
    """Game tree node for MCTS with neural network evaluation."""
    
    def __init__(self, start_state, nn_trainer, parent=None, state_cache={}):
        self.state = start_state
        self.children = None
        self.parent = parent
        self.nn_trainer = nn_trainer
        self.state_cache = state_cache
        
        if self.state not in self.state_cache:
            self.state_cache[self.state] = {
                "n": 0,
                "w": 0,
            }
    
    def value(self):
        """UCB1 value with neural network prior."""
        if self.state_cache[self.state]["n"] == 0:
            return np.inf
        
        # Exploration constant
        c = np.sqrt(2)
        
        # UCB1 formula: exploitation + exploration
        exploitation = self.state_cache[self.state]["w"] / self.state_cache[self.state]["n"]
        exploration = c * np.sqrt(np.log(self.state_cache[self.parent.state]["n"]) / self.state_cache[self.state]["n"])
        
        return exploitation + exploration
    
    def evaluate_with_nn(self):
        """Get the neural network's evaluation of the current state."""
        current_player = get_current_player(self.state)
        win_prob = self.nn_trainer.evaluate_position(self.state)
        
        # Convert the win probability to the perspective of the player to move
        if current_player == 1:  # Player 2
            win_prob = 1.0 - win_prob
            
        return win_prob
    
    def expand(self):
        """Expand this node in the game tree."""
        if is_terminal(self.state):
            # Terminal state - backpropagate actual winner
            self.backprop(check_winner(self.state))
        else:
            if self.children is None:
                # Leaf node - use neural network evaluation
                win_prob = self.evaluate_with_nn()
                self.backprop(win_prob)
                
                # Create children nodes
                possible_moves = access_cache_or_update(self.state)
                self.children = [
                    NNGameTree(
                        apply_move(self.state, a), 
                        self.nn_trainer, 
                        parent=self, 
                        state_cache=self.state_cache
                    ) for a in possible_moves
                ]
            else:
                # Non-leaf node - select child with highest UCB value
                values = [child.value() for child in self.children]
                self.children[np.argmax(values)].expand()
    
    def backprop(self, value):
        """Backpropagate the value through the tree."""
        self.state_cache[self.state]["n"] += 1
        
        if isinstance(value, (int, np.integer)):
            # Value is a winner (0 or 1)
            current_player = get_current_player(self.state)
            if value == current_player:
                # Current player won
                self.state_cache[self.state]["w"] += 1
        else:
            # Value is a win probability (float)
            self.state_cache[self.state]["w"] += value
        
        if self.parent is not None:
            # Continue backpropagation to parent
            self.parent.backprop(value)

    def find_child(self, state, depth=2):
        """Find a child node with the given state."""
        q = Queue()
        q.put((self, depth))

        while not q.empty():
            t, d = q.get()
            if t.state == state:
                return t
            if d > 0:
                if t.children is not None:
                    for child in t.children:
                        q.put((child, d-1))
        return None

class NNMCTSAgent:
    """Agent that uses MCTS with neural network evaluation."""
    
    def __init__(self, model_path=None, search_iters=100):
        self.search_iters = search_iters
        self.tree = None
        
        # Initialize neural network
        self.nn_trainer = QuoridorNNTrainer()
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.nn_trainer.load_model(model_path)
    
    def select_move(self, state):
        """Select the best move for the given state."""
        assert not is_terminal(state)
        
        # Initialize or update the game tree
        if self.tree is None:
            self.tree = NNGameTree(state, self.nn_trainer)
        elif self.tree.state != state:
            # Try to find existing child node
            child = self.tree.find_child(state)
            if child:
                self.tree = child
            else:
                # Create new tree with existing cache
                self.tree = NNGameTree(
                    state, 
                    self.nn_trainer, 
                    state_cache=self.tree.state_cache
                )
        
        # Run MCTS for the specified number of iterations
        for _ in range(self.search_iters):
            self.tree.expand()
        
        # Get possible moves and select the best one
        possible_moves = access_cache_or_update(state)
        cache = self.tree.state_cache
        
        # Select move with highest win rate
        best_idx = argmax(
            self.tree.children, 
            key=lambda tree: 0 if cache[tree.state]["n"] == 0 else cache[tree.state]["w"] / cache[tree.state]["n"]
        )
        
        return possible_moves[best_idx]
    
    def save_model(self, path):
        """Save the neural network model."""
        self.nn_trainer.save_model(path)


# Example usage
if __name__ == "__main__":
    from game_engine.game_state_tuple import create_initial_state, print_game_state
    
    # Create an agent
    agent = NNMCTSAgent(search_iters=100)
    
    # Create an initial state
    state = create_initial_state()
    
    # Select a move
    move = agent.select_move(state)
    print(f"Selected move: {move}")
    
    # Apply the move
    new_state = apply_move(state, move)
    
    # Print the game state
    print_game_state(new_state) 