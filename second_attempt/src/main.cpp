#include <torch/script.h>

#include <fstream>
#include <iostream>

#include "MCTS.hpp"

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  MCTS tree;

  Gamestate gs;

  Gamestate.module = torch::jit::load(argv[1]);

  while (!gs.terminal()) {
    tree.startNewSearch(gs);
    tree.iterate(10000);

    gs = tree.bestMoveApply();

    std::cout << "W " << tree.root->w << " N " << tree.root->n << std::endl;

    gs.displayBoard();

    std::cin.get();
  }
};