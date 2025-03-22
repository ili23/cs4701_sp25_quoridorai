#include "MCTS.hpp"

#include <iostream>

Move::Move(int x, int y) {
  pawnMove = true;
  pos = std::make_pair(x, y);
}

Move::Move(std::pair<int, int> p) {
  pawnMove = true;
  pos = p;
}

Move::Move() {}

Gamestate::Gamestate() {
  p1Pos.first = 4;
  p1Pos.second = 0;
  p2Pos.first = 4;
  p2Pos.second = 8;

  p1Turn = true;
}

void Gamestate::displayBoard() {
  std::cout << (p1Turn ? "X" : "O") << " to move" << std::endl;

  for (int y = kBoardSize - 1; y >= 0; y--) {
    std::cout << "  ";
    for (int x = 0; x < kBoardSize; x++) {
      // std::cout << " ";
      if (vFences[x][y] == 1) {
        std::cout << "<==>";
      } else {
        std::cout << "----";
      }
      std::cout << " ";
    }

    std::cout << std::endl;
    for (int x = 0; x < kBoardSize; x++) {
      if (x > 0 && hFences[x - 1][y] == 1) {
        std::cout << "[|]";
      } else {
        std::cout << " | ";
      }

      if (std::make_pair(x, y) == p1Pos) {
        std::cout << "X";
      } else if (std::make_pair(x, y) == p2Pos) {
        std::cout << "O";
      } else {
        std::cout << " ";
      }

      std::cout << " ";
      if (x == kBoardSize - 1) {
        std::cout << " | " << std::endl;
      }
    }
  }
  std::cout << "  ---- ---- ---- ---- ---- ---- ---- ---- ----" << std::endl;
}

std::unique_ptr<Gamestate> Gamestate::applyMove(const Move &m) const {
  std::unique_ptr<Gamestate> g = std::make_unique<Gamestate>(*this);

  if (g->p1Turn) {
    g->p1Pos = m.pos;
  } else {
    g->p2Pos = m.pos;
  }

  g->p1Turn = !g->p1Turn;

  return g;
}

bool inBounds(std::pair<int, int> x) {
  return x.first >= 0 && x.first < kBoardSize && x.second >= 0 &&
         x.second < kBoardSize;
}

std::vector<Move> Gamestate::getMoves() const {
  std::vector<Move> moves;

  std::pair<int, int> startingPoint = p1Turn ? p1Pos : p2Pos;
  std::pair<int, int> otherPawn = p1Turn ? p2Pos : p1Pos;

  std::pair<int, int> target =
      std::make_pair(startingPoint.first, startingPoint.second + 1);

  if (inBounds(target) && target != otherPawn &&
      !vFences[startingPoint.first][startingPoint.second]) {
    moves.emplace_back(target);
  }

  target = std::make_pair(startingPoint.first, startingPoint.second - 1);

  if (inBounds(target) && target != otherPawn &&
      !vFences[startingPoint.first][startingPoint.second - 1]) {
    moves.emplace_back(target);
  }

  target = std::make_pair(startingPoint.first + 1, startingPoint.second);

  if (inBounds(target) && target != otherPawn &&
      !hFences[startingPoint.first][startingPoint.second]) {
    moves.emplace_back(target);
  }

  target = std::make_pair(startingPoint.first - 1, startingPoint.second);

  if (inBounds(target) && target != otherPawn &&
      !hFences[startingPoint.first - 1][startingPoint.second]) {
    moves.emplace_back(target);
  }

  return moves;
}

void Gamestate::displayAllMoves() {
  std::vector<Move> allMoves = getMoves();

  for (Move m : allMoves) {
    auto v = applyMove(m);
    v->displayBoard();
  }
}