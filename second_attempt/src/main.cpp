#include <iostream>

#include "MCTS.hpp"

int main() {
  MCTS tree;

  Gamestate gs;

  while (!gs.terminal()) {
    tree.startNewSearch(gs);
    tree.iterate(10000);

    gs = tree.bestMoveApply();

    std::cout << "W " << tree.root->w << " N " << tree.root->n << std::endl;

    gs.displayBoard();

    std::cin.get();
  }
};