from libcpp.utility cimport pair
from libcpp.memory cimport shared_ptr

cdef extern from "MCTS.hpp":
    cdef cppclass Move:
        Move(int, int)
        Move()

    cdef cppclass Gamestate:
        Gamestate()
        void displayBoard()

        int hFences[9][9]
        int vFences[9][9]

        void displayAllMoves()

        shared_ptr[Gamestate]  applyMove(Move)

        