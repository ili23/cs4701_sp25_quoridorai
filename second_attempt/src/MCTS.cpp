#include "MCTS.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <cstdlib>   // For rand() and srand()
#include <ctime>     // For time()
#include <cmath>     // For std::exp

#include "Gamestate.hpp"

MCTS::MCTS() {
  // Seed the random number generator
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
};

void MCTS::startNewSearch(Gamestate &g) { root = std::make_unique<Node>(g); }

void MCTS::iterate(int n) {
  for (int i = 0; i < n; i++) {
    singleIterate();
  }
}

Gamestate MCTS::bestMoveApply() {
  // First, collect moves with their scores and visit counts
  std::vector<std::pair<Gamestate, float>> move_scores;
  std::vector<std::shared_ptr<Node>> move_nodes;


  float best_score = -std::numeric_limits<float>::infinity();
  Gamestate best_move;


  for (std::shared_ptr<Node> c : root->children) {
    if (c->n > 0) {
      float score = c->w / c->n;
      move_scores.emplace_back(c->state, score);
      move_nodes.push_back(c);


      if (score >= best_score) {
        best_move = c->state;
        best_score = score;
      }
    }
  }


 return best_move;

  
  // If no valid moves, return an empty state
  if (move_scores.empty()) {
    return Gamestate();
  }
  
  // Apply softmax to the scores to get a probability distribution
  std::vector<float> probabilities;
  
  // First, find the maximum score for numerical stability in softmax
  float max_score = -std::numeric_limits<float>::infinity();
  for (const auto& move : move_scores) {
    max_score = std::max(max_score, move.second);
  }
  
  // Calculate the softmax denominator
  float softmax_denom = 0.0f;
  for (const auto& move : move_scores) {
    softmax_denom += std::exp((move.second - max_score) * 5.0f); // Temperature parameter of 0.2 (1/5)
  }
  
  // Calculate the softmax probabilities
  for (const auto& move : move_scores) {
    float prob = std::exp((move.second - max_score) * 5.0f) / softmax_denom;
    probabilities.push_back(prob);
  }
  
  // Sample a move based on the distribution
  float random_val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  float cumulative_prob = 0.0;
  
  for (size_t i = 0; i < probabilities.size(); i++) {
    cumulative_prob += probabilities[i];
    if (random_val <= cumulative_prob) {
      root = move_nodes[i];
      return move_scores[i].first;
    }
  }
  
  // Fallback (should rarely happen due to floating point precision)
  root = move_nodes.back();
  return move_scores.back().first;
}

void MCTS::singleIterate() {
  // Step 1: Selection

  std::shared_ptr<Node> selection = root;

  while (!selection->leaf()) {
    float score = -std::numeric_limits<float>::infinity();
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
  float score = Gamestate::model_evaluate(selection->state);
  if (selection->state.p1Turn) {
    score = -1 * score;
  }

  // if (score != 0.5) {
  // std::cout << "evaluation is " << score << std::endl;
  // }

  // Step 4: Backpropogate
  selection->backpropogate(score);

  // std::cout << "backpropoage done. " << "  " << root->n << std::endl;
}