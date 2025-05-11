#include "Node.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include "Gamestate.hpp"

constexpr float c = 0;

Node::Node(Gamestate &g) {
  w = 0;
  n = 0;
  state = g;
  childrenExpanded = false;
  terminal = state.terminal();
  parent = nullptr;
}

Node::Node(Gamestate &g, std::shared_ptr<Node> p) {
  w = 0;
  n = 0;

  state = g;
  childrenExpanded = false;
  terminal = state.terminal();
  parent = p;
}

float Node::score() {
  if (n == 0) {
    std::cout << "Divide by zero!" << std::endl;
  }

  if (parent == nullptr) {
    return (w / n) + c * std::sqrt(std::log(n) / n);
  }
  return (w / n) + c * std::sqrt(std::log(parent->n) / n);
}

std::shared_ptr<Node> Node::expand() {
  if (terminal) {
    // std::cout << "Expanding from terminal" << std::endl;
    return shared_from_this();
  }

  if (childrenExpanded) {
    return children[0];
  }

  std::vector<Move> m = state.getMoves();
  for (Move move : m) {
    children.emplace_back(
        std::make_shared<Node>(*state.applyMove(move), shared_from_this()));
  }

  childrenExpanded = true;

  return children[0];
}

void Node::backpropogate(float score) {
  w += score;
  n += 1;

  if (parent != nullptr) {
    parent->backpropogate(-1 * score);
  }
}