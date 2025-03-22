import cython
from cython.cimports import MCTS


@cython.ccall
def testing():
    a = MCTS.Gamestate()
    a.hFences[4][4] = 1
    a.vFences[4][0] = 1
    a.displayAllMoves()
    # a.displayBoard()

    # move = MCTS.Move(4, 4)
    # b = cython.operator.dereference(a.applyMove(move))
    # b.displayBoard()
