#include <memory>
#include <utility>
#include <vector>

constexpr int kBoardSize = 9;

constexpr int kStartingFences = 10;

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
 public:
  Gamestate();

  std::pair<int, int> p1Pos;
  std::pair<int, int> p2Pos;

  int p1Fences = kStartingFences;
  int p2Fences = kStartingFences;

  // Horizontal fences are defined to be to the right of the space they are
  // listed on That is, a fence at (x, y) indicates that players can no longer
  // move between space (x, y) and (x + 1, y)
  int hFences[kBoardSize][kBoardSize] = {};

  // Vertical fences are defined to be above the space they are listed on.
  // That is, a fence at (x, y) indicates that players can no longer move
  // between space (x, y) and (x, y + 1)
  int vFences[kBoardSize][kBoardSize] = {};

  bool p1Turn;

  void displayBoard();

  void displayAllMoves();

  std::unique_ptr<Gamestate> applyMove(const Move &m) const;

  std::vector<Move> getMoves();
  bool pathToEnd(bool p1);

 private:
};
