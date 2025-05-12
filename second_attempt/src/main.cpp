#include <torch/script.h>

#include <fstream>
#include <iostream>
#include <vector>


#include "MCTS.hpp"

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  MCTS tree;

  Gamestate gs;

  try {
    Gamestate::module = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model: " << e.what() << std::endl;
    return -1;
  }

  std::vector<Gamestate> positions;

  while (!gs.terminal()) {
    tree.startNewSearch(gs);
    
    tree.iterate(10000);

    gs = tree.bestMoveApply();

    std::cout << "W " << tree.root->w << " N " << tree.root->n << std::endl;

    gs.displayBoard();

    positions.push_back(gs);
  }

  // Write positions to a csv
  std::ofstream myfile;
  myfile.open("gamestate.csv");

  for (auto gs : positions) {
    gs.write_csv(myfile);
  }

  myfile.close();

  return 0;
};