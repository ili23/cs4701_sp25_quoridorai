#include <fstream>
#include <iostream>
#include <vector>

#include "Constants.hpp"
#include "MCTS.hpp"

#ifdef TORCH
#include <torch/script.h>
#endif

int main(int argc, const char* argv[]) {
#ifdef TORCH
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
#endif

  int i = 0;

  int lines_written = 0;
  int file_num = 0;

  std::ofstream myfile;
  std::string path = "data/gamestate" + std::to_string(file_num) + ".csv";
  myfile.open(path);

  // Write column headers
  myfile << "player0_pawn,player1_pawn,num_walls_player0,num_walls_player1,"
            "move_count,current_player,";
  // Horizontal wall headers
  for (int i = 0; i < kBoardSize - 1; i++) {
    myfile << "h_wall_col" << i;
    myfile << ",";
  }

  // Vertical wall headers
  for (int i = 0; i < kBoardSize - 1; i++) {
    myfile << "v_wall_col" << i;
    myfile << ",";
  }

  myfile << "outcome" << std::endl;

  while (i < kGameCount) {
    MCTS tree;

    Gamestate gs;

#ifdef TORCH
    try {
      Gamestate::module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
      std::cerr << "Error loading the model: " << e.what() << std::endl;
      return -1;
    }
#endif

    std::vector<Gamestate> positions;

    int moves_made = 0;
    while (!gs.terminal()) {
      tree.startNewSearch(gs);

      if (moves_made < kRandomMovesCount) {
        gs = tree.randomMoveApply();
      } else {
        tree.iterate(kSearchIterations);

        if (moves_made % 2 == 0 && kPlayerInput) {
          int x;
          tree.root->expand();
          std::cout << "There are " << tree.root->children.size()
                    << " possible moves." << std::endl;

          for (size_t index = 0; index < tree.root->children.size(); index++) {
            std::cout << index << ". "
                      << tree.root->children[index]->m.to_string() << "\t\t"
                      << tree.root->children[index]->w << " "
                      << tree.root->children[index]->n << std::endl;
          }

          std::cin >> x;
          gs = tree.applyMove(x);
        } else {
          // std::cout << "==CONSIDERING=====" << std::endl;
          // tree.root->displayChildren();
          // std::cout << "==done CONSIDERING=====" << std::endl;

          gs = tree.bestMoveApply();
        }
      }
      
      std::cout << "Move: " << moves_made << std::endl;

      std::cout << "W " << tree.root->w << " N " << tree.root->n << std::endl;

      gs.displayBoard();

      if (moves_made >= kRandomMovesCount) {
        positions.push_back(gs);
      }

      moves_made++;
    }

    // Determine the winning player from the final game state
    int winning_player = -1;
    if (gs.terminal()) {
      float result = gs.result();
      if (result == 1.0) {
        winning_player = 0;  // Player 0 (p1) won
      } else if (result == -1.0) {
        winning_player = 1;  // Player 1 (p2) won
      } else {
        winning_player = 2;  // Draw
      }
    }

    // Write positions to a csv

    for (auto gs : positions) {
      gs.write_csv(myfile, winning_player);
    }

    lines_written += positions.size();

    // Open a new file
    if (lines_written > kMaxFileSize) {
      myfile.close();
      file_num++;
      lines_written = 0;
      path = "data/gamestate" + std::to_string(file_num) + ".csv";
      myfile.open(path);

      // Write column headers
      myfile << "player0_pawn,player1_pawn,num_walls_player0,num_walls_player1,"
                "move_count,current_player,";

      // Horizontal wall headers
      for (int i = 0; i < kBoardSize - 1; i++) {
        myfile << "h_wall_col" << i;
        myfile << ",";
      }

      // Vertical wall headers
      for (int i = 0; i < kBoardSize - 1; i++) {
        myfile << "v_wall_col" << i;
        myfile << ",";
      }
      myfile << "outcome" << std::endl;
    }

    i++;
  }

  if (myfile.is_open()) {
    myfile.close();
  }

  return 0;
};