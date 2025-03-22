import numpy as np
import random
from queue import Queue

from utility import argmax
from game_state_tuple import get_possible_moves, apply_move, is_terminal, check_winner, get_current_player



def rollout(state):
    t = 0
    while not is_terminal(state):
        a = random.choice(get_possible_moves(state))
        state = apply_move(state, a)
        t += 1
    print(f"rollout took {t} moves")
    return check_winner(state)

class GameTree:
    def __init__(self, start_state, parent=None):
        self.state = start_state
        self.possible_actions = get_possible_moves(self.state)
        self.children = None
        self.parent = parent
        self.n = 0
        self.w = 0

    def value(self):
        if self.n == 0:
            return np.inf
        c = np.sqrt(2)
        return (self.w / self.n) + c * np.sqrt(np.log(self.parent.n) / self.n)
    
    def expand(self):
        if is_terminal(self.state):
            self.backprop(check_winner(self.state))
        else:
            if self.children is None:
                self.backprop(rollout(self.state))
                self.children = [GameTree(apply_move(self.state, a), parent=self) for a in self.possible_actions]
            else:
                values = [child.value() for child in self.children]
                self.children[np.argmax(values)].expand()
    
    def backprop(self, winner):
        self.n += 1
        if winner is not None:  # There is a winner
            # The previous player is the opposite of current player in this state
            previous_player = 1 - get_current_player(self.state)
            # If the winner is the previous player, increment wins
            if previous_player == winner:
                self.w += 1
        
        if self.parent is not None:
            self.parent.backprop(winner)

    def find_child(self, state, depth=2):
        """"""
        q = Queue()
        q.put((self, depth))

        while q:
            t, d  = q.get()
            if t.state == state:
                return t
            if d > 0:
                if t.children is not None:
                    for child in t.children:
                        q.put((child, d-1))
        return None  # Return None if child not found

class Agent:
    def __init__(self, iters = 10):
        self.search_depth = iters
        self.tree = None
    
    def select_move(self, state, update_tree = True):
        assert not is_terminal(state)

        # Update tree so that root node has current board state
        if not (self.tree and update_tree):
            self.tree = GameTree(state)
        else:
            # If possible, look in old tree for cur state -> works well if the agent is playing a full game
            self.tree = self.tree.find_child(state)
            if self.tree is None:
                self.tree = GameTree(state)

        for _ in range(self.search_depth):
            self.tree.expand()

        return get_possible_moves(state)[argmax(self.tree.children, key=lambda tree: 0 if tree.n == 0 else tree.w / tree.n)]                
    

    