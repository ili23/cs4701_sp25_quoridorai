#include "Node.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include "Gamestate.hpp"

constexpr float c = 1.4142135;

Node::Node(Gamestate &g) {
  w = 0;
  n = 0;
  state = g;
  childrenExpanded = false;
  terminal = state.terminal();
}

Node::Node(Gamestate &g, std::shared_ptr<Node> p) {
  w = 0;
  n = 0;

  state = g;
  childrenExpanded = false;
  terminal = state.terminal();
  parent = p;
}

Node::Node(Gamestate &g, std::shared_ptr<Node> p, Move m_) {
  w = 0;
  n = 0;

  m = m_;

  state = g;
  childrenExpanded = false;
  terminal = state.terminal();
  parent = p;
}

void Node::displayChildren() {
  int idx = 0;
  for (auto c : children) {
    std::cout << idx << ". Possible move:" << std::endl;
    std::cout << "W " << c->w << " N " << c->n << std::endl;
    c->state.displayBoard();
    std::cout << std::endl << std::endl;
    idx++;
  }
}

float Node::score() {
  if (n == 0) {
    std::cout << "Divide by zero!" << std::endl;
  }

  if (auto sp = parent.lock()) {
    return (w / n) + c * std::sqrt(std::log(sp->n) / n);
  } else {
    return (w / n) + c * std::sqrt(std::log(n) / n);
  }
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
    children.emplace_back(std::make_shared<Node>(*state.applyMove(move),
                                                 shared_from_this(), move));
  }

  childrenExpanded = true;

  return children[0];
}

void Node::backpropogate(float score) {
  w += score;
  n += 1;

  if (auto sp = parent.lock()) {
    sp->backpropogate(-1 * score);
  }
}
