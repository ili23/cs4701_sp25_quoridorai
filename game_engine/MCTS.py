import numpy as np

def get_possible_moves(state):
    possible_moves = list("TODO")
    return possible_moves

def step(state, a):
    new_state = "TODO"
    return new_state

def isTerminal(state):
    """TODO return bool"""
    pass

def getWinner(state):
    """TODO return bool"""
    assert isTerminal(state)
    pass

def rollout(state):
    while not isTerminal(state):
        a = np.random.choice(get_possible_moves(state))
        state = step(state, a)
    
    return getWinner(state)

class GameState:
    def __init__(self, start_state, parent=None):
        self.state = start_state
        self.possible_actions = get_possible_moves(self.state)
        self.children = None
        self.parent = parent
        self.n = 0
        self.w = 0

    def value(self):
        c = np.sqrt(2)
        return (self.w / self.n) + c * np.sqrt(np.log(self.parent.n) / self.n)
    
    def expand(self):
        if self.state.is_terminal:
            self.backprop(getWinner(self.state))
        else:
            if self.children is None:
                self.backprop(rollout(self.state))
                self.children = [GameState(step(self.state, a), parent=self) for a in self.possible_actions]
            else:
                values = [child.value() for child in self.children]
                self.children[np.argmax(values)].expand()

    def backprop(self, winner):
        self.n += 1
        if self.state.current_player == winner:
            self.w += 1

        if self.parent is not None:
            self.backprop(winner)


start_state = "start state"
tree = GameState(start_state)
for i in range(1000):
    tree.expand()
    

    