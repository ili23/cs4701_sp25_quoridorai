import numpy as np
from queue import Queue

from utility import argmax

def isTerminal(state):
    """TODO return bool"""
    return state.winner is not None

def getWinner(state):
    """TODO return int"""
    assert isTerminal(state)
    return state.winner

def rollout(state):
    while not isTerminal(state):
        a = np.random.choice(state.get_possible_moves(state))
        state = state.step(state, a)
    
    return getWinner(state)

class GameTree:
    def __init__(self, start_state, parent=None):
        self.state = start_state
        self.possible_actions = self.state.get_possible_moves(self.state)
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
        if self.state.is_terminal:
            self.backprop(getWinner(self.state))
        else:
            if self.children is None:
                self.backprop(rollout(self.state))
                self.children = [GameTree(step(self.state, a), parent=self) for a in self.possible_actions]
            else:
                values = [child.value() for child in self.children]
                self.children[np.argmax(values)].expand()
    
    def backprop(self, winner):
        self.n += 1
        if self.state.current_player == winner:
            self.w += 1

        if self.parent is not None:
            self.backprop(winner)

    def find_child(self, state, depth=1):
        """"""
        q = Queue()
        q.put((self, depth))

        while q:
            t, d  = q.pop()
            if t.state == state:
                return t
            if d > 0:
                for child in t.children:
                    q.put((child, d-1))
        

class Agent:
    def __init__(self):
        self.search_depth
        self.tree = None
    
    def select_move(self, state, update_tree = True):
        assert not isTerminal(state)

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
    

    