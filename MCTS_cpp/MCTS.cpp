#include "MCTS.hpp"

#include <iostream>
#include <queue>

Move::Move(int x, int y) {
  pawnMove = true;
  pos = std::make_pair(x, y);
}

Move::Move(std::pair<int, int> p) {
  pawnMove = true;
  pos = p;
}

Move::Move(bool h, std::pair<int, int> p) {
  pawnMove = false;
  hFence = h;
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
        if (hFences[x][y]) {
          std::cout << "[|]" << std::endl;
        } else {
          std::cout << " | " << std::endl;
        }
      }
    }
  }
  std::cout << "  ---- ---- ---- ---- ---- ---- ---- ---- ----" << std::endl;
}

std::unique_ptr<Gamestate> Gamestate::applyMove(const Move &m) const {
  std::unique_ptr<Gamestate> g = std::make_unique<Gamestate>(*this);

  if (m.pawnMove) {
    if (g->p1Turn) {
      g->p1Pos = m.pos;
    } else {
      g->p2Pos = m.pos;
    }
  } else {
    if (g->p1Turn) {
      g->p1Fences--;
    } else {
      g->p2Fences--;
    }
    if (m.hFence) {
      g->hFences[m.pos.first][m.pos.second] = true;
    } else {
      g->vFences[m.pos.first][m.pos.second] = true;
    }
  }

  g->p1Turn = !g->p1Turn;

  return g;
}

bool inBounds(std::pair<int, int> x) {
  return x.first >= 0 && x.first < kBoardSize && x.second >= 0 &&
         x.second < kBoardSize;
}

std::vector<Move> Gamestate::getMoves() {
  std::vector<Move> moves;

  std::pair<int, int> startingPoint = p1Turn ? p1Pos : p2Pos;
  std::pair<int, int> otherPawn = p1Turn ? p2Pos : p1Pos;

  // Find valid pawn moves
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

  // Return if the current player has no more fences to place
  if ((p1Turn ? p1Fences : p2Fences) <= 0) {
    std::cout << "REUNGING EARLYT BC NUMBER OF FENCES " << std::endl;
    return moves;
  }

  // Find valid fence moves
  for (int x = 0; x < kBoardSize; x++) {
    for (int y = 0; y < kBoardSize; y++) {
      if (!hFences[x][y]) {
        hFences[x][y] = true;

        if (pathToEnd(true) && pathToEnd(false)) {
          moves.emplace_back(true, std::make_pair(x, y));
        } else {
          std::cout << "SKILPING FENCE BC NO MATH" << std::endl;
        }
        hFences[x][y] = false;
      }
      if (!vFences[x][y]) {
        vFences[x][y] = true;

        if (pathToEnd(true) && pathToEnd(false)) {
          moves.emplace_back(false, std::make_pair(x, y));
        }
        vFences[x][y] = false;
      }
    }
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

bool Gamestate::pathToEnd(bool p1) {
  bool reachable[kBoardSize][kBoardSize] = {};

  std::queue<std::pair<int, int>> toSearch;

  toSearch.push(p1 ? p1Pos : p2Pos);
  reachable[toSearch.front().first][toSearch.front().second] = true;

  std::pair<int, int> otherPawn = p1 ? p2Pos : p1Pos;

  std::pair<int, int> target;
  std::pair<int, int> startingPoint;

  while (!toSearch.empty()) {
    startingPoint = toSearch.front();

    target = std::make_pair(startingPoint.first, startingPoint.second + 1);

    if (inBounds(target) && !reachable[target.first][target.second] &&
        target != otherPawn &&
        !vFences[startingPoint.first][startingPoint.second]) {
      reachable[target.first][target.second] = true;
      toSearch.emplace(target);
    }

    target = std::make_pair(startingPoint.first, startingPoint.second - 1);

    if (inBounds(target) && !reachable[target.first][target.second] &&
        target != otherPawn &&
        !vFences[startingPoint.first][startingPoint.second - 1]) {
      reachable[target.first][target.second] = true;
      toSearch.emplace(target);
    }

    target = std::make_pair(startingPoint.first + 1, startingPoint.second);

    if (inBounds(target) && !reachable[target.first][target.second] &&
        target != otherPawn &&
        !hFences[startingPoint.first][startingPoint.second]) {
      reachable[target.first][target.second] = true;
      toSearch.emplace(target);
    }

    target = std::make_pair(startingPoint.first - 1, startingPoint.second);

    if (inBounds(target) && !reachable[target.first][target.second] &&
        target != otherPawn &&
        !hFences[startingPoint.first - 1][startingPoint.second]) {
      reachable[target.first][target.second] = true;
      toSearch.emplace(target);
    }

    toSearch.pop();
  }

  // Check back row
  // Player 1's back row is y = kBoardSize - 1
  // Player 2's back row is y = 0
  int y = p1 ? kBoardSize - 1 : 0;
  for (int x = 0; x < kBoardSize; x++) {
    if (reachable[x][y]) return true;
  }

  return false;
}