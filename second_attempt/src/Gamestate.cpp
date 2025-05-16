#include "Gamestate.hpp"

#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>

#include "model.hpp"

#ifdef TORCH
#include "torch/script.h"

torch::jit::script::Module Gamestate::module;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> game_state_to_tensors(
    Gamestate& state) {
  // Create a 4x17x17 tensor to represent the board state
  torch::Tensor board_tensor = torch::zeros({1, 4, 17, 17});

  // Set player positions based on whose turn it is
  if (state.p1Turn) {
    // Player 0 (current player) - always in channel 0
    int player0_row = state.p1Pos.first;
    int player0_col = state.p1Pos.second;
    board_tensor[0][0][player0_row][player0_col] = 1.0;

    // Player 1 (opponent) - always in channel 1
    int player1_row = state.p2Pos.first;
    int player1_col = state.p2Pos.second;
    board_tensor[0][1][player1_row][player1_col] = 1.0;

    // Set horizontal fences
    for (int row = 0; row < kBoardSize; row++) {
      for (int col = 0; col < kBoardSize; col++) {
        if (state.hFences[row][col]) {
          // Set fence in position and extend it horizontally
          board_tensor[0][2][row][col] = 1.0;
          board_tensor[0][2][row][col + 1] = 1.0;
          board_tensor[0][2][row][col + 2] = 1.0;
        }
      }
    }

    // Set vertical fences
    for (int row = 0; row < kBoardSize; row++) {
      for (int col = 0; col < kBoardSize; col++) {
        if (state.vFences[row][col]) {
          // Set fence in position and extend it vertically
          board_tensor[0][3][row][col] = 1.0;
          board_tensor[0][3][row + 1][col] = 1.0;
          board_tensor[0][3][row + 2][col] = 1.0;
        }
      }
    }
  } else {
    // If p2's turn, swap the perspective and flip the board

    // Player 0 (current player) - always in channel 0
    // Flip positions - for row/col we use (kBoardSize - 1 - pos)
    int player0_row = kBoardSize - 1 - state.p2Pos.first;
    int player0_col = kBoardSize - 1 - state.p2Pos.second;
    board_tensor[0][0][player0_row][player0_col] = 1.0;

    // Player 1 (opponent) - always in channel 1
    int player1_row = kBoardSize - 1 - state.p1Pos.first;
    int player1_col = kBoardSize - 1 - state.p1Pos.second;
    board_tensor[0][1][player1_row][player1_col] = 1.0;

    // Set horizontal fences - flip positions
    for (int row = 0; row < kBoardSize; row++) {
      for (int col = 0; col < kBoardSize; col++) {
        // When flipping the board, horizontal fences remain horizontal
        // but their position changes
        int flipped_row = kBoardSize - 1 - row;
        int flipped_col = kBoardSize - 1 - col - 2;  // Account for fence length

        if (flipped_col >= 0 && state.hFences[row][col]) {
          // Set fence in position and extend it horizontally
          board_tensor[0][2][flipped_row][flipped_col] = 1.0;
          board_tensor[0][2][flipped_row][flipped_col + 1] = 1.0;
          board_tensor[0][2][flipped_row][flipped_col + 2] = 1.0;
        }
      }
    }

    // Set vertical fences - flip positions
    for (int row = 0; row < kBoardSize; row++) {
      for (int col = 0; col < kBoardSize; col++) {
        // When flipping the board, vertical fences remain vertical
        // but their position changes
        int flipped_row = kBoardSize - 1 - row - 2;  // Account for fence length
        int flipped_col = kBoardSize - 1 - col;

        if (flipped_row >= 0 && state.vFences[row][col]) {
          // Set fence in position and extend it vertically
          board_tensor[0][3][flipped_row][flipped_col] = 1.0;
          board_tensor[0][3][flipped_row + 1][flipped_col] = 1.0;
          board_tensor[0][3][flipped_row + 2][flipped_col] = 1.0;
        }
      }
    }
  }

  // Create fence counts tensor - always from current player's perspective
  torch::Tensor fence_counts = torch::zeros({1, 2});
  if (state.p1Turn) {
    fence_counts[0][0] = state.p1Fences;  // Current player fences
    fence_counts[0][1] = state.p2Fences;  // Opponent fences
  } else {
    fence_counts[0][0] = state.p2Fences;  // Current player fences
    fence_counts[0][1] = state.p1Fences;  // Opponent fences
  }

  // Create move count tensor
  torch::Tensor move_count = torch::zeros({1, 1});
  move_count[0][0] = state.moveCount;

  return {board_tensor, fence_counts, move_count};
}
#endif

void Gamestate::write_csv(std::ofstream& f, int winning_player) {
  // Determine outcome based on winning_player
  // winning_player = 0 means player 0 (p1) won
  // winning_player = 1 means player 1 (p2) won

  float outcome;
  if (winning_player != 2) {
    outcome = (p1Turn ? (winning_player == 0 ? 1.0 : -1.0)
                      : (winning_player == 1 ? 1.0 : -1.0));
  } else {
    outcome = 0;
  }

  // Format:
  // player1_pawn,player2_pawn,num_walls_player1,num_walls_player2,move_count,current_player,
  // h_wall_col0-7,v_wall_col0-7,outcome

  // Player positions formatted as "row|col"
  f << p1Pos.first << "|" << p1Pos.second << ",";
  f << p2Pos.first << "|" << p2Pos.second << ",";

  // Number of walls for each player
  f << p1Fences << "," << p2Fences << ",";

  // Move count
  f << moveCount << ",";

  // Current player (0 for p1, 1 for p2)
  f << (p1Turn ? 0 : 1) << ",";

  // Horizontal walls for each column
  for (int col = 0; col < 8; col++) {
    // Format horizontal walls as pipe-separated values
    for (int row = 0; row < 8; row++) {
      // Add 1 if there's a wall, 0 otherwise
      if (row > 0) f << "|";  // Add pipe separator between values
      f << (row < kBoardSize - 1 && hFences[row][col] ? "1" : "0");
    }
    if (col < 7) f << ",";
  }

  // Add comma between h_walls and v_walls
  f << ",";

  // Vertical walls for each column
  for (int col = 0; col < 8; col++) {
    // Format vertical walls as pipe-separated values
    for (int row = 0; row < 8; row++) {
      // Add 1 if there's a wall, 0 otherwise
      if (row > 0) f << "|";  // Add pipe separator between values
      f << (row < kBoardSize - 1 && vFences[row][col] ? "1" : "0");
    }
    if (col < 7) f << ",";
  }

  // Write the outcome
  f << "," << outcome << std::endl;
}

using Edge = std::pair<int, int>;  // (fromNodeID, toNodeID)

// Returns shortest distance and path as list of edges using node IDs
std::pair<float, std::vector<Edge>> dijkstraRowWithPathIDs(
    const std::vector<std::vector<std::pair<int, int>>>& adj, int kBoardSize,
    int startRow, int startCol, bool targetBottom) {
  int totalNodes = kBoardSize * kBoardSize;
  std::vector<float> dist(totalNodes, std::numeric_limits<float>::infinity());
  std::vector<std::optional<int>> parent(totalNodes, std::nullopt);
  std::vector<bool> visited(totalNodes, false);

  using pii = std::pair<float, int>;  // (distance, nodeID)
  std::priority_queue<pii, std::vector<pii>, std::greater<pii>> pq;

  int start = startRow * kBoardSize + startCol;
  dist[start] = 0.0f;
  pq.push({0.0f, start});

  int end = -1;

  while (!pq.empty()) {
    auto [currDist, u] = pq.top();
    pq.pop();

    if (visited[u]) continue;
    visited[u] = true;

    int row = u / kBoardSize;
    if ((row == 0 && targetBottom) ||
        (row == kBoardSize - 1 && !targetBottom)) {
      end = u;
      break;
    }

    for (const auto& [v, weight] : adj[u]) {
      if (!visited[v] && dist[u] + weight < dist[v]) {
        dist[v] = dist[u] + weight;
        parent[v] = u;
        pq.push({dist[v], v});
      }
    }
  }

  std::vector<Edge> path;

  if (end != -1) {
    // Reconstruct path by backtracking from end to start
    int curr = end;
    while (parent[curr].has_value()) {
      int prev = parent[curr].value();
      path.emplace_back(prev, curr);
      curr = prev;
    }

    std::reverse(path.begin(), path.end());
  }

  return {end == -1 ? -1.0f : dist[end], path};
}

void addEdgeWeight(std::vector<std::vector<std::pair<int, int>>>& adj, int u,
                   int v, int newWeight) {
  // Update u -> v
  for (auto& p : adj[u]) {
    if (p.first == v) {
      p.second += newWeight;
      break;
    }
  }

  // Update v -> u (because the graph is undirected)
  for (auto& p : adj[v]) {
    if (p.first == u) {
      p.second += newWeight;
      break;
    }
  }
}

void printAdjacencyMatrix(
    const std::vector<std::vector<std::pair<int, int>>>& adj, int totalNodes) {
  std::vector<std::vector<int>> matrix(totalNodes,
                                       std::vector<int>(totalNodes, 0));

  // Fill matrix from adjacency list
  for (int u = 0; u < totalNodes; ++u) {
    for (const auto& [v, weight] : adj[u]) {
      matrix[u][v] = weight;
      // For undirected graphs, this is symmetric:
      matrix[v][u] = weight;
    }
  }

  // Print header
  std::cout << "   ";
  for (int j = 0; j < totalNodes; ++j) std::cout << std::setw(3) << j;
  std::cout << "\n";

  // Print each row
  for (int i = 0; i < totalNodes; ++i) {
    std::cout << std::setw(3) << i << " ";
    for (int j = 0; j < totalNodes; ++j) {
      std::cout << std::setw(3) << matrix[i][j];
    }
    std::cout << "\n";
  }
}

std::string displayNode(int nodeID) {
  int row = nodeID / kBoardSize;
  int col = nodeID % kBoardSize;

  return "(" + std::to_string(row) + ", " + std::to_string(col) + ")";
}

std::vector<std::vector<std::pair<int, int>>> subtractAdjacencyLists(
    const std::vector<std::vector<std::pair<int, int>>>& adj1,
    const std::vector<std::vector<std::pair<int, int>>>& adj2) {
  int n = adj1.size();
  std::vector<std::vector<std::pair<int, int>>> result(n);

  for (int u = 0; u < n; ++u) {
    // Build lookup map for adj2[u]
    std::unordered_map<int, int> adj2_map;
    for (const auto& [v, w2] : adj2[u]) {
      adj2_map[v] = w2;
    }

    for (const auto& [v, w1] : adj1[u]) {
      // Prevent duplicate undirected edges: only add u → v if u < v
      if (u < v) {
        int w2 = adj2_map.count(v) ? adj2_map[v] : 0;
        int w_diff = w1 - w2;

        // Optional: skip zero-weight results
        result[u].emplace_back(v, std::max(w_diff, 0));
        result[v].emplace_back(
            u, std::max(w_diff, 0));  // Undirected graph symmetry
      }
    }
  }

  return result;
}

double tree_eval(std::vector<double> features) {
  double features_mapped[8];

  for (int x = 0; x < kResilienceFeatureLength; x++) {
    if (std::isnan(features[0])) {
      features_mapped[x] = 0;
    } else {
      features_mapped[x] = (features[x] - means[x]) / scale[x];
    }
  }

  double prediction[3];
  score(features_mapped, prediction);

  return -1 * prediction[0] + 1 * prediction[2];
}

std::vector<double> Gamestate::resiliency_vector() {
  std::vector<double> result;

  int totalNodes = kBoardSize * kBoardSize;

  std::vector<std::vector<std::pair<int, int>>> adj1(totalNodes);
  std::vector<std::vector<std::pair<int, int>>> adj2(totalNodes);

  for (int r = 0; r < kBoardSize; ++r) {
    for (int c = 0; c < kBoardSize; ++c) {
      int u = r * kBoardSize + c;

      // Row = player pos first
      // Col = player pos second

      auto loc = std::make_pair(r, c);

      // Up
      if (r > 0 && !containsFenceInDirection(loc, -1, 0)) {
        int v = (r - 1) * kBoardSize + c;
        adj1[u].push_back({v, 1});
        adj1[v].push_back({u, 1});

        adj2[u].push_back({v, 1});
        adj2[v].push_back({u, 1});
      }
      // Right
      if (c < kBoardSize - 1 && !containsFenceInDirection(loc, 0, 1)) {
        int v = r * kBoardSize + (c + 1);
        adj1[u].push_back({v, 1});
        adj1[v].push_back({u, 1});

        adj2[u].push_back({v, 1});
        adj2[v].push_back({u, 1});
      }
      // Every edge is up or right of something
    }
  }

  int p2_weights_added = 0;
  for (int i = 0; i < kPathResilienceIters; i++) {
    auto [shortestDistance, pathEdges] = dijkstraRowWithPathIDs(
        adj2, kBoardSize, p2Pos.first, p2Pos.second, true);

    for (auto e : pathEdges) {
      addEdgeWeight(adj2, e.first, e.second, kPathResilienceWeight);
      p2_weights_added += kPathResilienceWeight;
    }
  }

  auto [p2FinalShortestDistance, pathEdges] =
      dijkstraRowWithPathIDs(adj2, kBoardSize, p2Pos.first, p2Pos.second, true);

  int p1_weights_added = 0;
  for (int i = 0; i < kPathResilienceIters; i++) {
    auto [shortestDistance, pathEdges] = dijkstraRowWithPathIDs(
        adj1, kBoardSize, p1Pos.first, p1Pos.second, false);

    for (auto e : pathEdges) {
      addEdgeWeight(adj1, e.first, e.second, kPathResilienceWeight);
      p1_weights_added += kPathResilienceWeight;
    }
  }

  auto [p1FinalShortestDistance, ed] = dijkstraRowWithPathIDs(
      adj1, kBoardSize, p1Pos.first, p1Pos.second, false);

  double p1_raw_resiliency =
      ((double)p1FinalShortestDistance) / kPathResilienceIters;
  double p1_scaled_resiliency =
      ((double)p1FinalShortestDistance) / p1_weights_added;

  double p2_raw_resiliency =
      ((double)p2FinalShortestDistance) / kPathResilienceIters;
  double p2_scaled_resiliency =
      ((double)p2FinalShortestDistance) / p2_weights_added;

  auto adj_p1liability = subtractAdjacencyLists(adj1, adj2);
  auto adj_p2liability = subtractAdjacencyLists(adj2, adj1);

  auto [p1LiabilityDistance, ed1] = dijkstraRowWithPathIDs(
      adj_p1liability, kBoardSize, p1Pos.first, p1Pos.second, false);

  auto [p2LiabilityDistance, ed2] = dijkstraRowWithPathIDs(
      adj_p2liability, kBoardSize, p2Pos.first, p2Pos.second, true);

  double p1_max_liability = 0;
  for (int u = 0; u < totalNodes; ++u) {
    for (const auto& [v, weight] : adj_p1liability[u]) {
      p1_max_liability = std::max(p1_max_liability, (double)weight);
    }
  }

  double p2_max_liability = 0;
  for (int u = 0; u < totalNodes; ++u) {
    for (const auto& [v, weight] : adj_p2liability[u]) {
      p2_max_liability = std::max(p2_max_liability, (double)weight);
    }
  }

  if (p1Turn) {
    return {p1_raw_resiliency,    p1_scaled_resiliency, p2_raw_resiliency,
            p2_scaled_resiliency, p1LiabilityDistance,  p2LiabilityDistance,
            p1_max_liability,     p2_max_liability};
  }

  return {p2_raw_resiliency,    p2_scaled_resiliency, p1_raw_resiliency,
          p1_scaled_resiliency, p2LiabilityDistance,  p1LiabilityDistance,
          p2_max_liability,     p1_max_liability};
}

float Gamestate::model_evaluate(Gamestate& g, bool smart_eval = false) {
#ifdef TORCH
  using namespace torch::indexing;
#endif

  if (g.terminal()) {
    return g.result();
  }

  if (!smart_eval) {
    return 0;
  }

  return tree_eval(g.resiliency_vector());

#ifdef TORCH
  // Use game_state_to_tensors to get all necessary tensors
  auto [board_tensor, fence_counts, move_count] = game_state_to_tensors(g);

  auto inner_tuple =
      torch::ivalue::Tuple::create({board_tensor, fence_counts, move_count});

  std::vector<torch::IValue> inputs;
  inputs.push_back(inner_tuple);

  torch::Tensor output = module.forward(inputs).toTensor();

  return output.item<float>();
#else
  int currentPlayerDistance, oppDistance;

  if (g.p1Turn) {
    currentPlayerDistance = kBoardSize - g.p1Pos.first - 1;
    oppDistance = g.p2Pos.first;
  } else {
    currentPlayerDistance = g.p2Pos.first;
    oppDistance = kBoardSize - g.p1Pos.first - 1;
  }

  // TODO: Remove this !p1Turn. Used to give different players different
  // heuristics
  if (currentPlayerDistance == oppDistance || !g.p1Turn) return 0;
  return (currentPlayerDistance < oppDistance ? 1 : -1) * 0.8 *
         ((float)std::abs(currentPlayerDistance - oppDistance)) /
         (kBoardSize - 2);
#endif
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

std::string Move::to_string() {
  if (pawnMove) {
    return "Pawn to " + std::to_string(pos.first) + ", " +
           std::to_string(pos.second);
  }

  if (hFence) {
    return "Horizontal fence at " + std::to_string(pos.first) + ", " +
           std::to_string(pos.second);
  }

  return "Vertical fence at " + std::to_string(pos.first) + ", " +
         std::to_string(pos.second);
}

Move::Move() {}

Gamestate::Gamestate() {
  p1Pos.first = p1_start_first;
  p1Pos.second = p1_start_second;
  p2Pos.first = p2_start_first;
  p2Pos.second = p2_start_second;

  p1Turn = true;

  // First Sample - histogram tactic
  hFences[1][1] = true;
  hFences[0][3] = true;
  vFences[2][0] = true;

  p1Pos.first = 2;
  p1Pos.second = 0;
  p2Pos.first = 2;
  p2Pos.second = 4;

  p1Turn = false;

  p1Fences = 0;
  p2Fences = 1;

  // Second Sample

  // p1Turn = true;
  // p1Pos.first = 0;
  // p1Pos.second = 1;

  // p2Pos.first = 3;
  // p2Pos.second = 2;

  // hFences[0][1] = true;
  // hFences[1][3] = true;
  // hFences[2][2] = true;
  // hFences[3][3] = true;
  // vFences[0][3] = true;
  // hFences[3][1] = true;
  // vFences[0][3] = true;
  // vFences[3][2] = true;
}

void Gamestate::displayBoard() {
  // ANSI color codes for better visualization
  const std::string BLUE = "\033[94m";   // For placed walls
  const std::string GRAY = "\033[90m";   // For potential wall slots
  const std::string GREEN = "\033[92m";  // For winner
  const std::string RESET = "\033[0m";   // Reset color

  std::cout << "Current board state:" << std::endl << std::endl;
  std::cout << "Evaluation is " << model_evaluate(*this) << std::endl;
  std::cout << "Smart Evaluation is " << model_evaluate(*this, true)
            << std::endl;

  if (terminal()) {
    if (moveCount >= kMaxMoves) {
      std::cout << GREEN << "Game ended in a tie after " << moveCount
                << " moves!" << RESET << std::endl;
    } else if (p1Pos.first == kBoardSize - 1) {
      std::cout << GREEN << "Player 0 has won the game!" << RESET << std::endl;
    } else if (p2Pos.first == 0) {
      std::cout << GREEN << "Player 1 has won the game!" << RESET << std::endl;
    }
  }

  // Pre-compute which grid points have fences passing through them
  bool hFenceExtended[kBoardSize][kBoardSize] = {};
  bool vFenceExtended[kBoardSize][kBoardSize] = {};

  // Mark horizontal fences (each fence is 2 units long)
  for (int row = 0; row < kBoardSize - 1; row++) {
    for (int col = 0; col < kBoardSize - 1; col++) {
      if (hFences[row][col]) {
        // No check is needed here because our fences matrix is always 1 unit
        // less than the board size
        hFenceExtended[row][col] = true;
        hFenceExtended[row][col + 1] = true;
      }
    }
  }

  // Mark vertical fences (each fence is 2 units long)
  for (int row = 0; row < kBoardSize - 1; row++) {
    for (int col = 0; col < kBoardSize - 1; col++) {
      if (vFences[row][col]) {
        // No check is needed here because our fences matrix is always 1 unit
        // less than the board size
        vFenceExtended[row][col] = true;
        vFenceExtended[row + 1][col] = true;
      }
    }
  }

  // Print the board with pawns and fences
  for (int row = 0; row < kBoardSize; row++) {
    // Print pawns and vertical fences
    for (int col = 0; col < kBoardSize; col++) {
      // Print pawn or empty space
      if (std::make_pair(row, col) == p1Pos) {
        std::cout << " 0 ";
      } else if (std::make_pair(row, col) == p2Pos) {
        std::cout << " 1 ";
      } else {
        std::cout << " . ";
      }

      // Print vertical fence if not at right edge
      if (col < kBoardSize - 1) {
        if (vFenceExtended[row][col]) {
          std::cout << BLUE << "║" << RESET;
        } else {
          std::cout << GRAY << "│" << RESET;
        }
      }
    }

    std::cout << std::endl;  // New line after row

    // Print horizontal fences if not at bottom edge
    if (row < kBoardSize - 1) {
      for (int col = 0; col < kBoardSize; col++) {
        if (hFenceExtended[row][col]) {
          std::cout << BLUE << "═══" << RESET;
        } else {
          std::cout << GRAY << "───" << RESET;
        }

        // Print intersection if not at right edge
        if (col < kBoardSize - 1) {
          std::cout << GRAY << "╬" << RESET;
        }
      }

      std::cout << std::endl;  // New line after horizontal fences
    }
  }

  std::cout << "Current player: " << (p1Turn ? "0" : "1") << std::endl;
  std::cout << "Player 0 fences left: " << p1Fences << std::endl;
  std::cout << "Player 1 fences left: " << p2Fences << std::endl;
  std::cout << "Moves played: " << moveCount << std::endl;

  if (terminal()) {
    if (moveCount >= kMaxMoves) {
      std::cout << GREEN << "Game ended in a tie after " << moveCount
                << " moves!" << RESET << std::endl;
    } else if (p1Pos.first == kBoardSize - 1) {
      std::cout << GREEN << "Player 0 has won the game!" << RESET << std::endl;
    } else if (p2Pos.first == 0) {
      std::cout << GREEN << "Player 1 has won the game!" << RESET << std::endl;
    }
  }

  std::cout << std::endl;
}

std::unique_ptr<Gamestate> Gamestate::applyMove(const Move& m) const {
  std::unique_ptr<Gamestate> g = std::make_unique<Gamestate>(*this);
  // prereq for this function is that the move is valid
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
  g->moveCount++;
  return g;
}

bool inBounds(std::pair<int, int> x) {
  return x.first >= 0 && x.first < kBoardSize && x.second >= 0 &&
         x.second < kBoardSize;
}

bool Gamestate::terminal() {
  return p1Pos.first == kBoardSize - 1 || p2Pos.first == 0 ||
         moveCount >= kMaxMoves;
}

float Gamestate::result() {
  if (moveCount >= kMaxMoves) {
    return 0;
  } else if (p1Pos.first == kBoardSize - 1)
    return 1;
  else if (p2Pos.first == 0)
    return -1;

  return 0;
}

std::vector<Move> Gamestate::getMoves() {
  std::vector<Move> moves;

  // Get pawn moves for current player
  std::vector<Move> pawnMoves = getPawnMoves();
  moves.insert(moves.end(), pawnMoves.begin(), pawnMoves.end());

  // Get fence moves for current player if they have fences remaining
  if ((p1Turn ? p1Fences : p2Fences) > 0) {
    std::vector<Move> fenceMoves = getFenceMoves();
    moves.insert(moves.end(), fenceMoves.begin(), fenceMoves.end());
  }

  return moves;
}

bool Gamestate::isValidBasicPawnMove(
    const std::pair<int, int>& startingPoint, int dx, int dy,
    const std::pair<int, int>& otherPawn) const {
  // Calculate target position
  std::pair<int, int> target = {startingPoint.first + dx,
                                startingPoint.second + dy};
  // Check if target is in bounds and not occupied by the other pawn
  if (!inBounds(target) || target == otherPawn) {
    return false;
  }

  // Case 1: Outside edges of the board
  bool onLeftVerticalEdge = startingPoint.second == 0;
  bool onRightVerticalEdge = startingPoint.second == kBoardSize - 1;
  bool onTopHorizontalEdge = startingPoint.first == 0;
  bool onBottomHorizontalEdge = startingPoint.first == kBoardSize - 1;
  bool isCorner = (onLeftVerticalEdge || onRightVerticalEdge) &&
                  (onTopHorizontalEdge || onBottomHorizontalEdge);

  if (isCorner) {
    int fenceX =
        onBottomHorizontalEdge ? startingPoint.first - 1 : startingPoint.first;
    int fenceY =
        onRightVerticalEdge ? startingPoint.second - 1 : startingPoint.second;
    if (dx != 0) {
      // moving vertically so check horizontal fence
      return !hFences[fenceX][fenceY];
    } else if (dy != 0) {
      // moving horizontally so check vertical fence
      return !vFences[fenceX][fenceY];
    }
  } else if (onLeftVerticalEdge) {
    if (dy != 0) {
      // moving horizontally so check vertical fence
      return !vFences[startingPoint.first - 1][startingPoint.second] &&
             !vFences[startingPoint.first][startingPoint.second];
    } else if (dx == -1) {
      // moving horizontally so check vertical fence
      return !hFences[startingPoint.first - 1][startingPoint.second];
    } else if (dx == 1) {
      // moving horizontally so check vertical fence
      return !hFences[startingPoint.first][startingPoint.second];
    }
  } else if (onRightVerticalEdge) {
    if (dy != 0) {
      // moving horizontally so check vertical fence
      return !vFences[startingPoint.first - 1][startingPoint.second - 1] &&
             !vFences[startingPoint.first][startingPoint.second - 1];
    } else if (dx == -1) {
      // moving vertically so check horizontal fence
      return !hFences[startingPoint.first - 1][startingPoint.second - 1];
    } else if (dx == 1) {
      // moving vertically so check horizontal fence
      return !hFences[startingPoint.first][startingPoint.second - 1];
    }
  } else if (onTopHorizontalEdge) {
    if (dx != 0) {
      // moving vertically so check horizontal fence
      return !hFences[startingPoint.first][startingPoint.second] &&
             !hFences[startingPoint.first][startingPoint.second - 1];
    } else if (dy == -1) {
      // moving horizontally so check vertical fence
      return !vFences[startingPoint.first][startingPoint.second - 1];
    } else if (dy == 1) {
      // moving horizontally so check vertical fence
      return !vFences[startingPoint.first][startingPoint.second];
    }
  } else if (onBottomHorizontalEdge) {
    if (dx != 0) {
      // moving vertically so check horizontal fence
      return !hFences[startingPoint.first - 1][startingPoint.second - 1] &&
             !hFences[startingPoint.first - 1][startingPoint.second];
    } else if (dy == -1) {
      // moving horizontally so check vertical fence
      return !vFences[startingPoint.first - 1][startingPoint.second - 1];
    } else if (dy == 1) {
      // moving horizontally so check vertical fence
      return !vFences[startingPoint.first - 1][startingPoint.second];
    }
  } else {  // Case 2: Starting is in a middle of the board
    if (dx == 1 && dy == 0) {
      // vertical move so check horizontal fences
      return !hFences[startingPoint.first][startingPoint.second - 1] &&
             !hFences[startingPoint.first][startingPoint.second];
    } else if (dx == -1 && dy == 0) {
      // vertical move so check horizontal fences
      return !hFences[startingPoint.first - 1][startingPoint.second - 1] &&
             !hFences[startingPoint.first - 1][startingPoint.second];
    } else if (dx == 0 && dy == 1) {
      // horizontal move so check vertical fences
      return !vFences[startingPoint.first - 1][startingPoint.second] &&
             !vFences[startingPoint.first][startingPoint.second];
    } else if (dx == 0 && dy == -1) {
      // horizontal move so check vertical fences
      return !vFences[startingPoint.first - 1][startingPoint.second - 1] &&
             !vFences[startingPoint.first][startingPoint.second - 1];
    }
  }
  assert(false);
  return false;
}

bool Gamestate::containsFenceInDirection(std::pair<int, int> startingPoint,
                                         int dx, int dy) {
  // Case 1: Outside edges of the board
  bool onLeftVerticalEdge = startingPoint.second == 0;
  bool onRightVerticalEdge = startingPoint.second == kBoardSize - 1;
  bool onTopHorizontalEdge = startingPoint.first == 0;
  bool onBottomHorizontalEdge = startingPoint.first == kBoardSize - 1;
  bool isCorner = (onLeftVerticalEdge || onRightVerticalEdge) &&
                  (onTopHorizontalEdge || onBottomHorizontalEdge);

  if (isCorner) {
    int fenceX =
        onBottomHorizontalEdge ? startingPoint.first - 1 : startingPoint.first;
    int fenceY =
        onRightVerticalEdge ? startingPoint.second - 1 : startingPoint.second;
    if (dx != 0) {
      // moving vertically so check horizontal fence
      return hFences[fenceX][fenceY];
    } else if (dy != 0) {
      // moving horizontally so check vertical fence
      return vFences[fenceX][fenceY];
    }
  } else if (onLeftVerticalEdge) {
    if (dy != 0) {
      // moving horizontally so check vertical fence
      return vFences[startingPoint.first - 1][startingPoint.second] ||
             vFences[startingPoint.first][startingPoint.second];
    } else if (dx == -1) {
      // moving horizontally so check vertical fence
      return hFences[startingPoint.first - 1][startingPoint.second];
    } else if (dx == 1) {
      // moving horizontally so check vertical fence
      return hFences[startingPoint.first][startingPoint.second];
    }
  } else if (onRightVerticalEdge) {
    if (dy != 0) {
      // moving horizontally so check vertical fence
      return vFences[startingPoint.first - 1][startingPoint.second - 1] ||
             vFences[startingPoint.first][startingPoint.second - 1];
    } else if (dx == -1) {
      // moving vertically so check horizontal fence
      return hFences[startingPoint.first - 1][startingPoint.second - 1];
    } else if (dx == 1) {
      // moving vertically so check horizontal fence
      return hFences[startingPoint.first][startingPoint.second - 1];
    }
  } else if (onTopHorizontalEdge) {
    if (dx != 0) {
      // moving vertically so check horizontal fence
      return hFences[startingPoint.first][startingPoint.second] ||
             hFences[startingPoint.first][startingPoint.second - 1];
    } else if (dy == -1) {
      // moving horizontally so check vertical fence
      return vFences[startingPoint.first][startingPoint.second - 1];
    } else if (dy == 1) {
      // moving horizontally so check vertical fence
      return vFences[startingPoint.first][startingPoint.second];
    }
  } else if (onBottomHorizontalEdge) {
    if (dx != 0) {
      // moving vertically so check horizontal fence
      return hFences[startingPoint.first - 1][startingPoint.second - 1] ||
             hFences[startingPoint.first - 1][startingPoint.second];
    } else if (dy == -1) {
      // moving horizontally so check vertical fence
      return vFences[startingPoint.first - 1][startingPoint.second - 1];
    } else if (dy == 1) {
      // moving horizontally so check vertical fence
      return vFences[startingPoint.first - 1][startingPoint.second];
    }
  } else {  // Case 2: Starting is in a middle of the board
    if (dx == 1 && dy == 0) {
      // vertical move so check horizontal fences
      return hFences[startingPoint.first][startingPoint.second - 1] ||
             hFences[startingPoint.first][startingPoint.second];
    } else if (dx == -1 && dy == 0) {
      // vertical move so check horizontal fences
      return hFences[startingPoint.first - 1][startingPoint.second - 1] ||
             hFences[startingPoint.first - 1][startingPoint.second];
    } else if (dx == 0 && dy == 1) {
      // horizontal move so check vertical fences
      return vFences[startingPoint.first - 1][startingPoint.second] ||
             vFences[startingPoint.first][startingPoint.second];
    } else if (dx == 0 && dy == -1) {
      // horizontal move so check vertical fences
      return vFences[startingPoint.first - 1][startingPoint.second - 1] ||
             vFences[startingPoint.first][startingPoint.second - 1];
    }
  }
  assert(false);  // should never reach here
}

std::vector<Move> Gamestate::getPawnMoves() {
  std::vector<Move> moves;

  std::pair<int, int> startingPoint = p1Turn ? p1Pos : p2Pos;
  std::pair<int, int> otherPawn = p1Turn ? p2Pos : p1Pos;

  // Basic directional moves (up, down, left, right)
  std::pair<int, int> directions[4] = {
      {0, 1},   // right
      {0, -1},  // left
      {1, 0},   // down
      {-1, 0}   // up
  };

  // Check all four basic directions
  for (const auto& dir : directions) {
    std::pair<int, int> target = {startingPoint.first + dir.first,
                                  startingPoint.second + dir.second};
    if (isValidBasicPawnMove(startingPoint, dir.first, dir.second, otherPawn)) {
      moves.emplace_back(target);
    }
  }

  // Check if opponent is adjacent
  int dx = otherPawn.first - startingPoint.first;
  int dy = otherPawn.second - startingPoint.second;

  if (abs(dx) + abs(dy) == 1) {
    if (containsFenceInDirection(
            startingPoint, dx,
            dy)) {  // Check for wall between player and opponent
      return moves;
    }

    // Try straight jump over opponent
    std::pair<int, int> jumpTarget =
        std::make_pair(otherPawn.first + dx, otherPawn.second + dy);

    bool straightJumpBlocked = false;

    if (inBounds(jumpTarget)) {  // check if jump target is in bounds
      if (containsFenceInDirection(
              otherPawn, dx, dy)) {  // check if there is a fence between the
                                     // opponent and the straight jump target
        straightJumpBlocked = true;
      } else {
        moves.emplace_back(jumpTarget);
        return moves;  // if the straight jump is valid, no diag jumps allowed
      }
    } else {
      // If jump target is out of bounds, consider it blocked
      straightJumpBlocked = true;
    }

    if (straightJumpBlocked) {
      std::vector<std::pair<int, int>> diagonals;

      // Add potential diagonal jumps (perpendicular to straight jump)
      if (dx == 0) {  // If moving horizontally, try vertical diagonals
        diagonals.push_back(
            std::make_pair(otherPawn.first + 1, otherPawn.second));
        diagonals.push_back(
            std::make_pair(otherPawn.first - 1, otherPawn.second));
      } else {  // If moving vertically, try horizontal diagonals
        diagonals.push_back(
            std::make_pair(otherPawn.first, otherPawn.second + 1));
        diagonals.push_back(
            std::make_pair(otherPawn.first, otherPawn.second - 1));
      }

      // Check each diagonal jump
      for (const auto& diagTarget : diagonals) {
        if (inBounds(diagTarget) && diagTarget != startingPoint) {
          if (!containsFenceInDirection(otherPawn,
                                        diagTarget.first - otherPawn.first,
                                        diagTarget.second - otherPawn.second)) {
            moves.emplace_back(diagTarget);
          }
        }
      }
    }
  }

  return moves;
}

std::vector<Move> Gamestate::getFenceMoves() {
  std::vector<Move> moves;

  // Return empty list if the current player has no more fences to place
  if ((p1Turn ? p1Fences : p2Fences) <= 0) {
    return moves;
  }

  // Find valid fence moves
  for (int row = 0; row < kBoardSize - 1; row++) {
    for (int col = 0; col < kBoardSize - 1; col++) {
      // Check horizontal fence placement
      if (!hFences[row][col]) {
        // Boundary check for horizontal fence (ensure it fits within the
        // board)

        bool isValid = true;

        // Check for adjacent fence that would conflict with 2-unit length
        if (isValid && col < kBoardSize - 2 && hFences[row][col + 1]) {
          isValid = false;  // Adjacent fence to right
        }

        if (isValid && col > 0 && hFences[row][col - 1]) {
          isValid = false;  // Adjacent fence to left extends to our position
        }

        // Check for crossing with vertical fence
        if (isValid && vFences[row][col]) {
          isValid = false;  // Crossing with fence at current position
        }

        // If valid placement, temporarily place fence and check paths
        if (isValid) {
          // Temporarily place fence
          hFences[row][col] = true;

          // Check if both players still have paths to goals
          if (pathToEnd(true) && pathToEnd(false)) {
            moves.emplace_back(true, std::make_pair(row, col));
          }

          // Remove temporary fence
          hFences[row][col] = false;
        }
      }

      // Check vertical fence placement
      if (!vFences[row][col]) {
        // Boundary check for vertical fence (ensure it fits within the board)
        bool isValid = true;

        // Check for adjacent fence that would conflict with 2-unit length
        if (isValid && row < kBoardSize - 2 && vFences[row + 1][col]) {
          isValid = false;  // Adjacent fence below
        }

        if (isValid && row > 0 && vFences[row - 1][col]) {
          isValid = false;  // Adjacent fence above extends to our position
        }

        // Check for crossing with horizontal fence
        if (isValid && hFences[row][col]) {
          isValid = false;  // Crossing with fence at current position
        }

        // If valid placement, temporarily place fence and check paths
        if (isValid) {
          // Temporarily place fence
          vFences[row][col] = true;
          // Print the board state for debugging
          // Check if both players still have paths to goals
          if (pathToEnd(true) && pathToEnd(false)) {
            moves.emplace_back(false, std::make_pair(row, col));
          }
          // Remove temporary fence
          vFences[row][col] = false;
        }
      }
    }
  }

  return moves;
}

bool Gamestate::pathToEnd(bool p1) {
  bool reachable[kBoardSize][kBoardSize] = {};

  std::queue<std::pair<int, int>> toSearch;

  // Start at current player position
  std::pair<int, int> startPos = p1 ? p1Pos : p2Pos;

  toSearch.push(startPos);
  reachable[startPos.first][startPos.second] = true;

  std::pair<int, int> directions[4] = {
      {0, 1},   // right
      {0, -1},  // left
      {1, 0},   // down
      {-1, 0}   // up
  };

  while (!toSearch.empty()) {
    std::pair<int, int> currentPos = toSearch.front();
    toSearch.pop();

    // Check all four basic directions
    for (const auto& dir : directions) {
      int dx = dir.first;
      int dy = dir.second;
      std::pair<int, int> target = {currentPos.first + dx,
                                    currentPos.second + dy};

      // Check if the move is within bounds and not already visited
      if (!inBounds(target) || reachable[target.first][target.second]) {
        continue;
      }

      bool blocked = false;

      // Check for fences based on movement direction
      blocked = containsFenceInDirection(currentPos, dx, dy);

      if (!blocked) {
        reachable[target.first][target.second] = true;
        toSearch.emplace(target);
      }
    }
  }

  // Check if the player reached their goal row
  // Player 0's goal is bottom row (row = BOARD_SIZE - 1)
  // Player 1's goal is top row (row = 0)
  int goalRow = p1 ? kBoardSize - 1 : 0;

  // Check if any position in the goal row is reachable
  for (int col = 0; col < kBoardSize; col++) {
    if (reachable[goalRow][col]) return true;
  }

  return false;
}
