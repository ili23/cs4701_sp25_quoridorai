import numpy as np
import random
from queue import Queue

from utility import argmax
from game_state_tuple import get_possible_moves, apply_move, is_terminal, check_winner, get_current_player

POSSIBLE_MOVES = {}
def access_cache_or_update(state):
    if state not in POSSIBLE_MOVES:
        POSSIBLE_MOVES[state] = get_possible_moves(state)
    return POSSIBLE_MOVES[state]

def rollout(state, max_steps=100):
    t = 0
    while not is_terminal(state) and t < max_steps:
        # Heuristic: prefer moves that advance toward goal
        moves = access_cache_or_update(state)
        current_player = get_current_player(state)
        
        # Prioritize pawn moves that advance toward goal
        pawn_moves = [m for m in moves if m[0] == "move"]
        if pawn_moves:
            goal_row = 8 if current_player == 0 else 0
            # Sort moves by distance to goal
            pawn_moves.sort(key=lambda m: abs(m[1][0] - goal_row))
            # 70% chance to pick advancing move, 30% random
            if random.random() < 1.0:
                a = pawn_moves[0]  # Choose move closest to goal
            else:
                a = random.choice(moves)
        else:
            a = random.choice(moves)
            
        state = apply_move(state, a)
        t += 1
    print(f"rollout took {t} moves")
    return check_winner(state)

class GameTree:
    def __init__(self, start_state, parent=None, state_cache={}):
        self.state = start_state
        self.children = None
        self.parent = parent
        self.state_cache = state_cache
        if self.state not in self.state_cache:
            self.state_cache[self.state] = {
                "n": 0,
                "w": 0,
            }
        
    def value(self):
        if self.state_cache[self.state]["n"] == 0:
            return np.inf
        c = np.sqrt(2)
        return (self.state_cache[self.state]["w"] / self.state_cache[self.state]["n"]) + c * np.sqrt(np.log(self.state_cache[self.parent.state]["n"]) / self.state_cache[self.state]["n"])
    
    def expand(self):
        if is_terminal(self.state):
            self.backprop(check_winner(self.state))
        else:
            if self.children is None:
                self.backprop(rollout(self.state))
                self.children = [GameTree(apply_move(self.state, a), parent=self, state_cache=self.state_cache) for a in access_cache_or_update(self.state)]
            else:
                values = [child.value() for child in self.children]
                self.children[np.argmax(values)].expand()
    
    def backprop(self, winner):
        self.state_cache[self.state]["n"] += 1
        if winner is not None:  # There is a winner
            # The previous player is the opposite of current player in this state
            previous_player = 1 - get_current_player(self.state)
            # If the winner is the previous player, increment wins
            if previous_player == winner:
                self.state_cache[self.state]["w"] += 1
        
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
    def __init__(self, iters = 100):
        self.search_depth = iters
        self.tree = None
    
    def select_move(self, state):
        assert not is_terminal(state)
        if self.tree is None: 
            self.tree = GameTree(state)
        elif self.tree.state != state:
            self.tree = GameTree(state, state_cache=self.tree.state_cache)

        for _ in range(self.search_depth):
            self.tree.expand()
        
        possible_moves = access_cache_or_update(state)
        cache = self.tree.state_cache

        return possible_moves[argmax(self.tree.children, key=lambda tree: 0 if cache[tree.state]["n"] == 0 else cache[tree.state]["w"] / cache[tree.state]["n"])]          
    
    