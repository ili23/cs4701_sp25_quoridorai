#include <memory>

#include "Gamestate.hpp"
#include "Node.hpp"

class MCTS {
 public:
  MCTS();

  void startNewSearch(Gamestate &g);

  void iterate(int n);

  Gamestate applyMove(int);

  Gamestate bestMove();

  Gamestate bestMoveApply();

  Gamestate randomMoveApply();

  std::shared_ptr<Node> root;

 private:
  void backpropagate(std::shared_ptr<Node> finalNode, float result);

  void singleIterate();
};
