#include "MCTS.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>

#include "Gamestate.hpp"

MCTS::MCTS() {};

void MCTS::startNewSearch(Gamestate &g) { root = std::make_unique<Node>(g); }

void MCTS::iterate(int n) {
  for (int i = 0; i < n; i++) {
    singleIterate();
  }
}

Gamestate MCTS::bestMove() {
  Gamestate b;
  float score = std::numeric_limits<float>::min();

  for (std::shared_ptr<Node> c : root->children) {
    if (c->n > 0 && c->w / c->n >= score) {
      b = c->state;

      score = c->w / c->n;
    }
  }

  return b;
}

Gamestate MCTS::bestMoveApply() {
  Gamestate b;
  float score = std::numeric_limits<float>::min();

  std::shared_ptr<Node> new_root;
  for (std::shared_ptr<Node> c : root->children) {
    if (c->n > 0 && c->w / c->n >= score) {
      b = c->state;

      // std::cout << "Considering " << std::endl;
      // b.displayBoard();
      // std::cout << c->w << "  " << c->n << std::endl;

      std::cout << "in btestMovePPlaqy if statment" << std::endl;

      score = c->w / c->n;
      new_root = c;
    }
  }
  root = new_root;
  return b;
}

void MCTS::singleIterate() {
  // Step 1: Selection

  std::shared_ptr<Node> selection = root;

  while (!selection->leaf()) {
    float score = std::numeric_limits<float>::min();
    std::shared_ptr<Node> best_child = nullptr;

    for (std::shared_ptr<Node> c : selection->children) {
      if (!c->childrenExpanded || c->n == 0) {
        best_child = c;
        break;
      }

      float candidate_score = c->score();

      if (best_child == nullptr || candidate_score >= score) {
        score = candidate_score;
        best_child = c;
      }
    }

    selection = best_child;
  }

  // std::cout << "Selection done." << std::endl;

  // Step 2: Expansion
  selection = selection->expand();

  // if (selection->state.terminal()) {
  //   std::cout << "looking at a temrinal state!" << std::endl;
  // }

  // std::cout << "expansion done." << std::endl;

  // Step 3: Simulation
  float score = evaluate(selection->state);
  if (selection->state.p1Turn) {
    score = 1 - score;
  }

  // if (score != 0.5) {
  // std::cout << "evaluation is " << score << std::endl;
  // }

  // Step 4: Backpropogate
  selection->backpropogate(score);

  // std::cout << "backpropoage done. " << "  " << root->n << std::endl;
}