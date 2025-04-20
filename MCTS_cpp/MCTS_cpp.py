import cython
from cython.cimports import MCTS


@cython.ccall
def testing():
    a = MCTS.Gamestate()
    a.hFences[4][4] = 1
    a.vFences[4][0] = 1
    for x in range(9):
        a.vFences[x][4] = 1

    a.vFences[4][4] = 0

    # print(f"The path to the end is: {a.pathToEnd(0)}")

    # a.vFences[4][4] = 1

    # print(f"However, after adding the fence, that path to the end is: {a.pathToEnd(0)}")

    a.displayAllMoves()
    # a.displayBoard()

    # move = MCTS.Move(4, 4)
    # b = cython.operator.dereference(a.applyMove(move))
    # b.displayBoard()
