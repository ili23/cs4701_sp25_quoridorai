#ifndef MCTS_HPP_
#define MCTS_HPP_

#include <memory>
#include <vector>
#include <unordered_map>
#include <random>
#include <cmath>
#include <queue>
#include <algorithm>
#include <functional>
#include <limits>

#include "game_state_tuple.hpp"

// Forward declarations
class Move;
class Gamestate;

// State cache struct to store visit counts and wins
struct StateInfo {
    int n = 0;  // visit count
    int w = 0;  // win count
};

// GameTree class for MCTS
class GameTree {
public:
    GameTree(const Gamestate& state, GameTree* parent = nullptr, 
             std::unordered_map<Gamestate, StateInfo>* stateCache = nullptr);
    ~GameTree();

    double value() const;
    void expand();
    void backprop(int winner);
    GameTree* findChild(const Gamestate& state, int depth = 2);

    std::vector<std::unique_ptr<GameTree>> children;
    Gamestate state;
    GameTree* parent;
    std::unordered_map<Gamestate, StateInfo>* stateCache;
};

// Agent class that uses MCTS to select moves
class Agent {
public:
    Agent(int iterations = 100);
    ~Agent();

    Move selectMove(const Gamestate& state);

private:
    int searchDepth;
    std::unique_ptr<GameTree> tree;
    std::unordered_map<Gamestate, StateInfo> stateCache;
};

// Helper functions
bool isTerminal(const Gamestate& state);
int checkWinner(const Gamestate& state);
bool getCurrentPlayer(const Gamestate& state);

// MCTS specific functions
int rollout(const Gamestate& state, int maxSteps = 100);
std::vector<Move> accessCacheOrUpdate(const Gamestate& state);

#endif // MCTS_HPP_ 