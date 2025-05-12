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

  // Determine the winning player from the final game state
  int winning_player = -1;
  if (gs.terminal()) {
    float result = gs.result();
    if (result == 1.0) {
      winning_player = 0; // Player 0 (p1) won
    } else if (result == -1.0) {
      winning_player = 1; // Player 1 (p2) won
    } else {
      winning_player = 2; // Draw
    }
  }

  // Write positions to a csv
  std::ofstream myfile;
  myfile.open("gamestate.csv");

  // Write column headers
  myfile << "player0_pawn,player1_pawn,num_walls_player0,num_walls_player1,move_count,current_player,";
  
  // Horizontal wall headers
  for (int i = 0; i < 8; i++) {
    myfile << "h_wall_col" << i;
    if (i < 7) myfile << ",";
  }
  
  myfile << ",";
  
  // Vertical wall headers
  for (int i = 0; i < 8; i++) {
    myfile << "v_wall_col" << i;
    if (i < 7) myfile << ",";
  }
  
  myfile << ",outcome" << std::endl;

  for (auto gs : positions) {
    gs.write_csv(myfile, winning_player);
  }

  myfile.close();

  return 0;
};