#include "Gamestate.hpp"

#include <torch/script.h>

#include <iostream>
#include <queue>

torch::jit::script::Module Gamestate::module;

float Gamestate::model_evaluate(Gamestate& g) {
  using namespace torch::indexing;

  if (g.terminal()) {
    return g.result() == 1 ? 1 : -1;
  }

  torch::Tensor board_tensor = torch::zeros({1, 4, 17, 17});
  torch::Tensor fence_counts = torch::zeros({1, 2});
  torch::Tensor move_count = torch::zeros({1, 1});

  fence_counts.index_put_({0, 0}, g.p1Fences);
  fence_counts.index_put_({0, 1}, g.p2Fences);

  // FILL IN HERE

  auto inner_tuple =
      torch::ivalue::Tuple::create({board_tensor, fence_counts, move_count});

  std::vector<torch::IValue> inputs;
  inputs.push_back(inner_tuple);

  torch::Tensor output = module.forward(inputs).toTensor();

  return output.item<float>();
}

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
  p1Pos.first = 6;
  p1Pos.second = 5;
  p2Pos.first = 8;
  p2Pos.second = 0;

  p1Turn = true;
}

void Gamestate::displayBoard() {
  std::cout << (p1Turn ? "0" : "1") << " to move" << std::endl;
  std::cout << "P1 is at " << p1Pos.first << " " << p1Pos.second << std::endl;
  std::cout << "P2 is at " << p2Pos.first << " " << p2Pos.second << std::endl;

  std::cout << "Evaluation is " << model_evaluate(*this) << std::endl;

  for (int y = kBoardSize - 1; y >= 0; y--) {
    std::cout << "  ";
    for (int x = 0; x < kBoardSize; x++) {
      if (vFences[x][y]) {
        std::cout << "<==>";
      } else {
        std::cout << "----";
      }
      std::cout << " ";
    }

    std::cout << std::endl;
    for (int x = 0; x < kBoardSize; x++) {
      if (x > 0 && hFences[x - 1][y]) {
        std::cout << "[|]";
      } else {
        std::cout << " | ";
      }

      if (std::make_pair(x, y) == p1Pos) {
        std::cout << "0";
      } else if (std::make_pair(x, y) == p2Pos) {
        std::cout << "1";
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

std::unique_ptr<Gamestate> Gamestate::applyMove(const Move& m) const {
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

bool Gamestate::terminal() {
  return p1Pos.first == kBoardSize - 1 || p2Pos.first == 0;
}

float Gamestate::result() {
  if (p1Pos.first == kBoardSize - 1)
    return 1;
  else if (p2Pos.first == 0)
    return 0;

  return 0.5;
}

std::vector<Move> Gamestate::getMoves() {
  std::vector<Move> moves;

  std::pair<int, int> startingPoint = p1Turn ? p1Pos : p2Pos;
  std::pair<int, int> otherPawn = p1Turn ? p2Pos : p1Pos;

  // Basic directional moves (up, down, left, right)
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

  // Jump moves over opponent (matching Python implementation)
  // Check if opponent is adjacent
  int dx = otherPawn.first - startingPoint.first;
  int dy = otherPawn.second - startingPoint.second;

  if (abs(dx) + abs(dy) == 1) {  // Adjacent pawns
    // Direction from player to opponent
    // Try straight jump over opponent
    std::pair<int, int> jumpTarget =
        std::make_pair(otherPawn.first + dx, otherPawn.second + dy);

    // Check if jump target is valid
    if (inBounds(jumpTarget)) {
      bool canJump = false;

      // Check fence between opponent and jump target
      if (dx == 0) {   // Vertical jump
        if (dy > 0) {  // Jumping up
          canJump = !vFences[otherPawn.first][otherPawn.second];
        } else {  // Jumping down
          canJump = !vFences[otherPawn.first][otherPawn.second - 1];
        }
      } else {         // Horizontal jump
        if (dx > 0) {  // Jumping right
          canJump = !hFences[otherPawn.first][otherPawn.second];
        } else {  // Jumping left
          canJump = !hFences[otherPawn.first - 1][otherPawn.second];
        }
      }

      if (canJump) {
        moves.emplace_back(jumpTarget);
      } else {
        // If straight jump is blocked, try diagonal jumps
        std::vector<std::pair<int, int>> diagonals;

        // Add potential diagonal jumps (perpendicular to straight jump)
        if (dx == 0) {  // If moving vertically, try horizontal diagonals
          diagonals.push_back(
              std::make_pair(otherPawn.first + 1, otherPawn.second));
          diagonals.push_back(
              std::make_pair(otherPawn.first - 1, otherPawn.second));
        } else {  // If moving horizontally, try vertical diagonals
          diagonals.push_back(
              std::make_pair(otherPawn.first, otherPawn.second + 1));
          diagonals.push_back(
              std::make_pair(otherPawn.first, otherPawn.second - 1));
        }

        // Check each diagonal jump
        for (const auto& diagTarget : diagonals) {
          if (inBounds(diagTarget) && diagTarget != startingPoint) {
            bool canDiagJump = false;

            // Check fence between opponent and diagonal target
            if (diagTarget.first == otherPawn.first + 1) {  // Jumping right
              canDiagJump = !hFences[otherPawn.first][otherPawn.second];
            } else if (diagTarget.first ==
                       otherPawn.first - 1) {  // Jumping left
              canDiagJump = !hFences[otherPawn.first - 1][otherPawn.second];
            } else if (diagTarget.second ==
                       otherPawn.second + 1) {  // Jumping up
              canDiagJump = !vFences[otherPawn.first][otherPawn.second];
            } else if (diagTarget.second ==
                       otherPawn.second - 1) {  // Jumping down
              canDiagJump = !vFences[otherPawn.first][otherPawn.second - 1];
            }

            if (canDiagJump) {
              moves.emplace_back(diagTarget);
            }
          }
        }
      }
    }
  }

  // Return if the current player has no more fences to place
  if ((p1Turn ? p1Fences : p2Fences) <= 0) {
    return moves;
  }

  // Find valid fence moves
  for (int x = 0; x < kBoardSize - 1; x++) {
    for (int y = 0; y < kBoardSize - 1; y++) {
      // Only consider valid fence placements where one more fence won't block
      // paths
      if (!hFences[x][y]) {
        // Check if horizontal fence would cross with vertical fence
        bool wouldCross = false;

        // Check for crossing with vertical fence at this position or next
        // position
        if (vFences[x][y] || (x < kBoardSize - 2 && vFences[x + 1][y])) {
          wouldCross = true;
        }

        // Check for adjacent fence that would conflict with 2-unit length
        bool conflictLength = false;
        if ((x > 0 && hFences[x - 1][y]) ||
            (x < kBoardSize - 2 && hFences[x + 1][y])) {
          conflictLength = true;
        }

        if (!wouldCross && !conflictLength) {
          // Temporarily place fence to check path to end
          hFences[x][y] = true;

          if (pathToEnd(true) && pathToEnd(false)) {
            moves.emplace_back(true, std::make_pair(x, y));
          }

          // Remove temporary fence
          hFences[x][y] = false;
        }
      }

      if (!vFences[x][y]) {
        // Check if vertical fence would cross with horizontal fence
        bool wouldCross = false;

        // Check for crossing with horizontal fence at this position or next
        // position
        if (hFences[x][y] || (y < kBoardSize - 2 && hFences[x][y + 1])) {
          wouldCross = true;
        }

        // Check for adjacent fence that would conflict with 2-unit length
        bool conflictLength = false;
        if ((y > 0 && vFences[x][y - 1]) ||
            (y < kBoardSize - 2 && vFences[x][y + 1])) {
          conflictLength = true;
        }

        if (!wouldCross && !conflictLength) {
          // Temporarily place fence to check path to end
          vFences[x][y] = true;

          if (pathToEnd(true) && pathToEnd(false)) {
            moves.emplace_back(false, std::make_pair(x, y));
          }

          // Remove temporary fence
          vFences[x][y] = false;
        }
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

  // Start at current player position
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

  // Check if the player reached their goal row
  // Player 0's goal is bottom row (row = BOARD_SIZE - 1)
  // Player 1's goal is top row (row = 0)
  int goalRow = p1 ? kBoardSize - 1 : 0;

  // Check if any position in the goal row is reachable
  for (int x = 0; x < kBoardSize; x++) {
    if (reachable[x][goalRow]) return true;
  }

  return false;
}
