#pragma once

#ifdef TORCH
#include <torch/script.h>
#endif

#include <fstream>
#include <functional>  // For std::hash
#include <memory>
#include <utility>
#include <vector>

#include "Constants.hpp"

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

  std::string to_string();
};

class Gamestate {
 public:
  Gamestate();

  void write_csv(std::ofstream& f, int winning_player);

#ifdef TORCH
  static torch::jit::script::Module module;
#endif

  static float model_evaluate(Gamestate& g);

  std::pair<int, int> p1Pos;
  std::pair<int, int> p2Pos;

  int p1Fences = kStartingFences;
  int p2Fences = kStartingFences;
  int moveCount = 0;
  // Horizontal fences are defined to be to the right of the space they are
  // listed on That is, a fence at (x, y) indicates that players can no longer
  // move between space (x, y) and (x + 1, y)
  bool hFences[kBoardSize-1][kBoardSize-1] = {};

  // Vertical fences are defined to be above the space they are listed on.
  // That is, a fence at (x, y) indicates that players can no longer move
  // between space (x, y) and (x, y + 1)
  bool vFences[kBoardSize-1][kBoardSize-1] = {};

  bool p1Turn;  // True if p1 is the current player, false if p2 is the current
                // player

  void displayBoard();

  std::unique_ptr<Gamestate> applyMove(const Move& m) const;

  // Get all valid moves for the current player
  std::vector<Move> getMoves();
  
  // Get valid pawn moves for the current player
  std::vector<Move> getPawnMoves();
  
  // Get valid fence placement moves for the current player
  std::vector<Move> getFenceMoves();

  bool pathToEnd(bool p1);

  // Checks if a basic pawn move in a given direction is valid
  bool isValidBasicPawnMove(const std::pair<int, int>& startingPoint, int dx, int dy, const std::pair<int, int>& otherPawn) const;

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
           std::equal(&hFences[0][0], &hFences[0][0] + (kBoardSize-1) * (kBoardSize-1),
                      &other.hFences[0][0]) &&
           std::equal(&vFences[0][0], &vFences[0][0] + (kBoardSize-1) * (kBoardSize-1),
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
    for (int i = 0; i < kBoardSize-1; i++) {
      for (int j = 0; j < kBoardSize-1; j++) {
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