#include "MCTS.hpp"

#include <chrono>
#include <iostream>
#include <random>

// Global variables
std::unordered_map<Gamestate, std::vector<Move>> POSSIBLE_MOVES;
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

// Helper function implementations
bool isTerminal(const Gamestate& state) { return checkWi nner(state) != -1; }

int checkWinner(const Gamestate& state) {
  // Player 1 wins by reaching the bottom row
  if (state.p1Pos.second == kBoardSize - 1) {
    return 0;
  }

  // Player 2 wins by reaching the top row
  if (state.p2Pos.second == 0) {
    return 1;
  }

  // No winner yet
  return -1;
}

bool getCurrentPlayer(const Gamestate& state) { return state.p1Turn; }

std::vector<Move> accessCacheOrUpdate(Gamestate& state) {
  if (POSSIBLE_MOVES.find(state) == POSSIBLE_MOVES.end()) {
    POSSIBLE_MOVES[state] = state.getMoves();
  }
  return POSSIBLE_MOVES[state];
}

int rollout(const Gamestate& state, int maxSteps) {
  Gamestate currentState = state;
  int t = 0;

  while (!isTerminal(currentState) && t < maxSteps) {
    // Get all possible moves
    std::vector<Move> moves = accessCacheOrUpdate(currentState);
    if (moves.empty()) {
      break;
    }

    // Collect pawn moves for heuristic selection
    std::vector<Move> pawnMoves;
    for (const auto& move : moves) {
      if (move.pawnMove) {
        pawnMoves.push_back(move);
      }
    }

    Move selectedMove;
    if (!pawnMoves.empty()) {
      // Current player
      bool currentPlayer = getCurrentPlayer(currentState);
      int goalRow = currentPlayer ? kBoardSize - 1 : 0;

      // Sort moves by distance to goal
      std::sort(pawnMoves.begin(), pawnMoves.end(),
                [goalRow](const Move& a, const Move& b) {
                  return std::abs(a.pos.second - goalRow) <
                         std::abs(b.pos.second - goalRow);
                });

      // Use the move closest to the goal
      selectedMove = pawnMoves[0];
    } else {
      // Randomly select a move
      std::uniform_int_distribution<int> dist(0, moves.size() - 1);
      selectedMove = moves[dist(rng)];
    }

    // Apply the selected move
    currentState = *currentState.applyMove(selectedMove);
    t++;
  }

  std::cout << "rollout took " << t << " moves" << std::endl;
  return checkWinner(currentState);
}

// GameTree implementation
GameTree::GameTree(const Gamestate& state, GameTree* parent,
                   std::unordered_map<Gamestate, StateInfo>* stateCache)
    : state(state), parent(parent), stateCache(stateCache) {
  // Initialize the state entry if it doesn't exist
  if (stateCache != nullptr && stateCache->find(state) == stateCache->end()) {
    (*stateCache)[state] = StateInfo();
  }
}

GameTree::~GameTree() {
  // Children are managed by unique_ptr, so they'll be deleted automatically
}

double GameTree::value() const {
  if ((*stateCache)[state].n == 0) {
    return std::numeric_limits<double>::infinity();
  }

  const double c = std::sqrt(2.0);
  return (double)(*stateCache)[state].w / (*stateCache)[state].n +
         c * std::sqrt(std::log((*stateCache)[parent->state].n) /
                       (*stateCache)[state].n);
}

void GameTree::expand() {
  if (isTerminal(state)) {
    backprop(checkWinner(state));
  } else {
    if (children.empty()) {
      backprop(rollout(state));

      // Create children for all possible moves
      std::vector<Move> moves = accessCacheOrUpdate(state);
      for (const auto& move : moves) {
        std::unique_ptr<Gamestate> nextState = state.applyMove(move);
        children.push_back(
            std::make_unique<GameTree>(*nextState, this, stateCache));
      }
    } else {
      // Find the child with the highest UCB value
      double maxValue = -std::numeric_limits<double>::infinity();
      size_t bestChildIndex = 0;

      for (size_t i = 0; i < children.size(); i++) {
        double childValue = children[i]->value();
        if (childValue > maxValue) {
          maxValue = childValue;
          bestChildIndex = i;
        }
      }

      // Expand the best child
      children[bestChildIndex]->expand();
    }
  }
}

void GameTree::backprop(int winner) {
  (*stateCache)[state].n += 1;

  if (winner != -1) {  // There is a winner
    // The previous player is the opposite of current player in this state
    bool previousPlayer = !getCurrentPlayer(state);

    // If the winner is the previous player, increment wins
    if ((previousPlayer && winner == 0) || (!previousPlayer && winner == 1)) {
      (*stateCache)[state].w += 1;
    }
  }

  if (parent != nullptr) {
    parent->backprop(winner);
  }
}

GameTree* GameTree::findChild(const Gamestate& targetState, int depth) {
  std::queue<std::pair<GameTree*, int>> q;
  q.push({this, depth});

  while (!q.empty()) {
    auto [tree, d] = q.front();
    q.pop();

    if (tree->state.p1Pos == targetState.p1Pos &&
        tree->state.p2Pos == targetState.p2Pos &&
        tree->state.p1Turn == targetState.p1Turn) {
      return tree;
    }

    if (d > 0 && !tree->children.empty()) {
      for (const auto& child : tree->children) {
        q.push({child.get(), d - 1});
      }
    }
  }

  return nullptr;  // Child not found
}

// Agent implementation
Agent::Agent(int iterations) : searchDepth(iterations), tree(nullptr) {}

Agent::~Agent() {
  // Tree is managed by unique_ptr, so it'll be deleted automatically
}

Move Agent::selectMove(Gamestate& state) {
  // Make sure the state is not terminal
  if (isTerminal(state)) {
    throw std::runtime_error("Cannot select a move for a terminal state");
  }

  // Create or update the tree
  if (tree == nullptr) {
    tree = std::make_unique<GameTree>(state, nullptr, &stateCache);
  } else if (tree->state.p1Pos != state.p1Pos ||
             tree->state.p2Pos != state.p2Pos ||
             tree->state.p1Turn != state.p1Turn) {
    // If the state has changed, create a new tree with the existing state cache
    tree = std::make_unique<GameTree>(state, nullptr, &stateCache);
  }

  // Run the search algorithm
  for (int i = 0; i < searchDepth; i++) {
    tree->expand();
  }

  // Get all possible moves
  std::vector<Move> possibleMoves = accessCacheOrUpdate(state);

  // Find the child with the best win ratio
  double bestScore = -1.0;
  size_t bestChildIndex = 0;

  for (size_t i = 0; i < tree->children.size(); i++) {
    const auto& childState = tree->children[i]->state;

    // Only consider if the state has been visited
    if (stateCache[childState].n > 0) {
      double score =
          (double)stateCache[childState].w / stateCache[childState].n;

      if (score > bestScore) {
        bestScore = score;
        bestChildIndex = i;
      }
    }
  }

  // Find which move leads to the best child state
  const auto& bestChildState = tree->children[bestChildIndex]->state;

  // Match the move from possible moves that leads to this state
  for (size_t i = 0; i < possibleMoves.size(); i++) {
    std::unique_ptr<Gamestate> nextState = state.applyMove(possibleMoves[i]);

    if (nextState->p1Pos == bestChildState.p1Pos &&
        nextState->p2Pos == bestChildState.p2Pos) {
      return possibleMoves[i];
    }
  }

  // Fallback to first move if somehow no match is found
  return possibleMoves[0];
}