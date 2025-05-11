#include <memory>

#include "Gamestate.hpp"
#include "Node.hpp"

class MCTS {
 public:
  MCTS();

  void startNewSearch(Gamestate &g);

  void iterate(int n);

  void applyMove(Move m);

  Gamestate bestMove();

  Gamestate bestMoveApply();

  std::shared_ptr<Node> root;

 private:
  void backpropagate(std::shared_ptr<Node> finalNode, float result);

  void singleIterate();
};
