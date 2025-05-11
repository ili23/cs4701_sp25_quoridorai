#include "MCTS.hpp"

#include <chrono>
#include <iostream>
#include <random>

// Global variables
std::unordered_map<Gamestate, std::vector<Move>> POSSIBLE_MOVES;
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

// Helper function implementations
bool isTerminal(const Gamestate& state) { return checkWinner(state) != -1; }

int checkWinner(const Gamestate& state) {
  // Player 0 wins by reaching the bottom row (row = BOARD_SIZE - 1)
  if (state.p1Pos.first == kBoardSize - 1) {
    return 0;
  }

  // Player 1 wins by reaching the top row (row = 0)
  if (state.p2Pos.first == 0) {
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

// Function to validate fence placements don't trap players
bool isValidFencePlacement(const Gamestate& state, const Move& move) {
  // Skip validation for pawn moves
  if (move.pawnMove) {
    return true;
  }

  // Apply the move to get the next state
  std::unique_ptr<Gamestate> nextState = state.applyMove(move);

  // Check if both players have a path to their respective goals
  bool p1HasPath = nextState->pathToEnd(true);   // Player 0 to bottom row
  bool p2HasPath = nextState->pathToEnd(false);  // Player 1 to top row

  // Both players must have a path to their goals
  return p1HasPath && p2HasPath;
}

int rollout(const Gamestate& state, int maxSteps) {
  Gamestate currentState = state;
  int t = 0;
  int maxAttempts =
      10;  // Limit attempts to find valid move to avoid infinite loops

  while (!isTerminal(currentState) && t < maxSteps) {
    // Get all possible moves
    std::vector<Move> moves = accessCacheOrUpdate(currentState);
    if (moves.empty()) {
      break;
    }

    // Pre-filter moves to remove invalid fence placements
    std::vector<Move> validMoves;
    for (const auto& move : moves) {
      if (move.pawnMove || isValidFencePlacement(currentState, move)) {
        validMoves.push_back(move);
      }
    }

    // If no valid moves after filtering, this shouldn't happen in a valid game
    if (validMoves.empty()) {
      // Fall back to pawn moves only
      for (const auto& move : moves) {
        if (move.pawnMove) {
          validMoves.push_back(move);
        }
      }

      // If still empty, something is wrong - break out
      if (validMoves.empty()) {
        std::cerr << "Warning: No valid moves found in rollout!" << std::endl;
        break;
      }
    }

    // Separate pawn moves and fence moves
    std::vector<Move> pawnMoves;
    std::vector<Move> fenceMoves;

    for (const auto& move : validMoves) {
      if (move.pawnMove) {
        pawnMoves.push_back(move);
      } else {
        fenceMoves.push_back(move);
      }
    }

    Move selectedMove;

    // Check if we have pawn moves
    if (!pawnMoves.empty()) {
      // 70% of the time, choose a pawn move towards the goal
      std::uniform_real_distribution<double> probDist(0.0, 1.0);
      bool useHeuristic = probDist(rng) < 0.7;

      if (useHeuristic) {
        // Current player
        bool currentPlayer = getCurrentPlayer(currentState);
        int goalRow = currentPlayer ? kBoardSize - 1 : 0;

        // Sort moves by distance to goal
        std::sort(pawnMoves.begin(), pawnMoves.end(),
                  [goalRow](const Move& a, const Move& b) {
                    return std::abs(a.pos.first - goalRow) <
                           std::abs(b.pos.first - goalRow);
                  });

        // Use the move closest to the goal
        selectedMove = pawnMoves[0];
      } else {
        // Randomly select a pawn move
        std::uniform_int_distribution<int> dist(0, pawnMoves.size() - 1);
        selectedMove = pawnMoves[dist(rng)];
      }
    } else if (!fenceMoves.empty()) {
      // Randomly select a fence move - all should be valid at this point
      std::uniform_int_distribution<int> dist(0, fenceMoves.size() - 1);
      selectedMove = fenceMoves[dist(rng)];
    } else {
      // Randomly select any move if neither pawn nor fence moves were found
      std::uniform_int_distribution<int> dist(0, validMoves.size() - 1);
      selectedMove = validMoves[dist(rng)];
    }

    // Apply the selected move
    currentState = *currentState.applyMove(selectedMove);
    t++;
  }

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

  // Standard UCB1 formula
  const double c = 1.414;  // sqrt(2)
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
        // Only create children for valid moves
        if (move.pawnMove || isValidFencePlacement(state, move)) {
          std::unique_ptr<Gamestate> nextState = state.applyMove(move);
          children.push_back(
              std::make_unique<GameTree>(*nextState, this, stateCache));
        }
      }

      // If no valid moves were found (shouldn't happen), add all pawn moves
      if (children.empty()) {
        for (const auto& move : moves) {
          if (move.pawnMove) {
            std::unique_ptr<Gamestate> nextState = state.applyMove(move);
            children.push_back(
                std::make_unique<GameTree>(*nextState, this, stateCache));
          }
        }
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
    // TODO: FIX THIS; needs to alternate somehow
    parent->backprop(winner);
  }
}

GameTree* GameTree::findChild(const Gamestate& targetState, int depth) {
  std::queue<std::pair<GameTree*, int>> q;
  q.push({this, depth});

  while (!q.empty()) {
    auto [tree, d] = q.front();
    q.pop();

    // Compare all aspects of the game state to find a match
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

  // Pre-filter to remove invalid fence placements
  std::vector<Move> validMoves;
  for (const auto& move : possibleMoves) {
    if (move.pawnMove || isValidFencePlacement(state, move)) {
      validMoves.push_back(move);
    }
  }

  // If no valid moves after filtering (shouldn't happen), use pawn moves only
  if (validMoves.empty()) {
    for (const auto& move : possibleMoves) {
      if (move.pawnMove) {
        validMoves.push_back(move);
      }
    }

    // If still empty, something is wrong
    if (validMoves.empty()) {
      throw std::runtime_error("No valid moves found after filtering");
    }
  }

  // Find the child with the best win ratio
  double bestScore = -1.0;
  size_t bestChildIndex = 0;
  bool foundValidChild = false;

  for (size_t i = 0; i < tree->children.size(); i++) {
    const auto& childState = tree->children[i]->state;

    // Only consider if the state has been visited
    if (stateCache[childState].n > 0) {
      double score =
          (double)stateCache[childState].w / stateCache[childState].n;

      if (score > bestScore) {
        bestScore = score;
        bestChildIndex = i;
        foundValidChild = true;
      }
    }
  }

  if (foundValidChild) {
    // Find which move leads to the best child state
    const auto& bestChildState = tree->children[bestChildIndex]->state;

    // Match the move from valid moves that leads to this state
    for (size_t i = 0; i < validMoves.size(); i++) {
      std::unique_ptr<Gamestate> nextState = state.applyMove(validMoves[i]);

      if (nextState->p1Pos == bestChildState.p1Pos &&
          nextState->p2Pos == bestChildState.p2Pos &&
          nextState->p1Turn == bestChildState.p1Turn) {
        return validMoves[i];
      }
    }
  }

  // Fallback: If no valid move found through MCTS, use first valid move
  return validMoves[0];
}