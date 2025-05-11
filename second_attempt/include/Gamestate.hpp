#pragma once

#include <torch/script.h>

#include <functional>  // For std::hash
#include <memory>
#include <utility>
#include <vector>

constexpr int kBoardSize = 9;

constexpr int kStartingFences = 0;

class Move {
 public:
  // Move the current pawn to the given position, (x, y)
  Move(int x, int y);
  Move(std::pair<int, int> p);

  // Create a fence placing move. If h is true, place a horizontal fence,
  // otherwise place a vertical fence
  Move(bool h, std::pair<int, int> p);
  Move();

  std::pair<int, int> pos;
  bool pawnMove;
  bool hFence;
};

class Gamestate {
  static torch::jit::script::Module module;

  static float evaluate(Gamestate& g);

 public:
  Gamestate();

  std::pair<int, int> p1Pos;
  std::pair<int, int> p2Pos;

  int p1Fences = kStartingFences;
  int p2Fences = kStartingFences;

  // Horizontal fences are defined to be to the right of the space they are
  // listed on That is, a fence at (x, y) indicates that players can no longer
  // move between space (x, y) and (x + 1, y)
  bool hFences[kBoardSize][kBoardSize] = {};

  // Vertical fences are defined to be above the space they are listed on.
  // That is, a fence at (x, y) indicates that players can no longer move
  // between space (x, y) and (x, y + 1)
  bool vFences[kBoardSize][kBoardSize] = {};

  bool p1Turn;

  void displayBoard();

  void displayAllMoves();

  std::unique_ptr<Gamestate> applyMove(const Move& m) const;

  std::vector<Move> getMoves();
  bool pathToEnd(bool p1);

  // Return true if the game is over
  bool terminal();

  // Return 1 if p1 has won, 0 if p2 has won and 0.5 otherwise
  float result();

  // Equality operator for unordered_map
  bool operator==(const Gamestate& other) const {
    return p1Pos == other.p1Pos && p2Pos == other.p2Pos &&
           p1Fences == other.p1Fences && p2Fences == other.p2Fences &&
           p1Turn == other.p1Turn &&
           // Compare fence arrays
           std::equal(&hFences[0][0], &hFences[0][0] + kBoardSize * kBoardSize,
                      &other.hFences[0][0]) &&
           std::equal(&vFences[0][0], &vFences[0][0] + kBoardSize * kBoardSize,
                      &other.vFences[0][0]);
  }

 private:
};

// Hash function for Gamestate to allow it to be used as a key in unordered_map
namespace std {
template <>
struct hash<Gamestate> {
  std::size_t operator()(const Gamestate& state) const {
    std::size_t seed = 0;

    // Hash the positions
    seed ^=
        hash<int>()(state.p1Pos.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hash<int>()(state.p1Pos.second) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^=
        hash<int>()(state.p2Pos.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hash<int>()(state.p2Pos.second) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);

    // Hash the fence counts
    seed ^=
        hash<int>()(state.p1Fences) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^=
        hash<int>()(state.p2Fences) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

    // Hash the turn
    seed ^= hash<bool>()(state.p1Turn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

    // Hash the fence arrays (simplified for performance)
    for (int i = 0; i < kBoardSize; i++) {
      for (int j = 0; j < kBoardSize; j++) {
        seed ^= hash<bool>()(state.hFences[i][j]) + 0x9e3779b9 + (seed << 6) +
                (seed >> 2);
        seed ^= hash<bool>()(state.vFences[i][j]) + 0x9e3779b9 + (seed << 6) +
                (seed >> 2);
      }
    }

    return seed;
  }
};
}  // namespace std

float evaluate(Gamestate& g);