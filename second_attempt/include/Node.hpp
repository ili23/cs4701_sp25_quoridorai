#include <memory>
#include <vector>

#include "Gamestate.hpp"

class Node : public std::enable_shared_from_this<Node> {
 public:
  Node(Gamestate &g);
  Node(Gamestate &g, std::shared_ptr<Node> parent);

  Gamestate state;

  float w, n;

  std::shared_ptr<Node> parent;

  bool leaf() { return !childrenExpanded || terminal; };

  std::vector<std::shared_ptr<Node>> children;

  std::shared_ptr<Node> expand();

  float score();

  void backpropogate(float score);

  bool childrenExpanded, terminal;
};