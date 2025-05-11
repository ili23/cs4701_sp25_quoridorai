#include <iostream>

#include "MCTS.hpp"
#include <torch/script.h>
#include <fstream>


int main() {
  MCTS tree;

  Gamestate gs;

  std::cout << "start" << std::endl;
  
  std::ifstream test("model.pt");
  if (!test.good()) {
      std::cerr << "File not found or unreadable!" << std::endl;
  }
  else{
      std::cerr << "File found" << std::endl;
  }
  
  torch::jit::script::Module module;
  try {
    std::string model_path = "model.pt";
    std::cout << "Trying to load: [" << model_path << "]" << std::endl;
    module = torch::jit::load(model_path);
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n" << e.what() << std::endl;
    return -1;
  }

  std::cout << "after" << std::endl;
  
  while (!gs.terminal()) {
    tree.startNewSearch(gs);
    tree.iterate(10000);

    gs = tree.bestMoveApply();

    std::cout << "W " << tree.root->w << " N " << tree.root->n << std::endl;

    gs.displayBoard();

    std::cin.get();
  }
};