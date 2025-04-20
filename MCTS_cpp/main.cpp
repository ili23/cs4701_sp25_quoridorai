#include <chrono>
#include <iostream>

#include "MCTS.hpp"

int main() {
  // Create an initial gamestate
  Gamestate initialState;
  std::cout << "Initial board state:" << std::endl;
  initialState.displayBoard();
  
  // Test pathToEnd function
  std::cout << "Player 0 has path to end: " << (initialState.pathToEnd(true) ? "true" : "false") << std::endl;
  std::cout << "Player 1 has path to end: " << (initialState.pathToEnd(false) ? "true" : "false") << std::endl;
  
  // Display all valid moves
  std::cout << "Calculating all valid moves..." << std::endl;
  std::vector<Move> validMoves = initialState.getMoves();
  std::cout << "Number of valid moves: " << validMoves.size() << std::endl;
  
  // Count pawn moves and fence moves
  int pawnMoves = 0;
  int hFenceMoves = 0;
  int vFenceMoves = 0;
  for (const auto& move : validMoves) {
    if (move.pawnMove) {
      pawnMoves++;
    } else if (move.hFence) {
      hFenceMoves++;
    } else {
      vFenceMoves++;
    }
  }
  std::cout << "Pawn moves: " << pawnMoves << std::endl;
  std::cout << "Horizontal fence moves: " << hFenceMoves << std::endl;
  std::cout << "Vertical fence moves: " << vFenceMoves << std::endl;

  // Create MCTS agent
  Agent agent(100);  // 100 iterations

  // Game loop
  Gamestate currentState = initialState;
  int turn = 0;

  while (!isTerminal(currentState) && turn < 30) {
    std::cout << "\nTurn " << turn + 1 << ":" << std::endl;
    std::cout << "Current player: " << (currentState.p1Turn ? "0" : "1") << std::endl;
    std::cout << "Player 0 position: (" << currentState.p1Pos.first << "," << currentState.p1Pos.second << ")" << std::endl;
    std::cout << "Player 1 position: (" << currentState.p2Pos.first << "," << currentState.p2Pos.second << ")" << std::endl;

    // Measure time for move selection
    auto start = std::chrono::high_resolution_clock::now();

    // Select a move using MCTS
    Move selectedMove = agent.selectMove(currentState);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Report the move
    std::cout << "Player " << (currentState.p1Turn ? "0" : "1");
    if (selectedMove.pawnMove) {
      std::cout << " moves pawn to (" << selectedMove.pos.first << ", "
                << selectedMove.pos.second << ")";
    } else {
      std::cout << " places "
                << (selectedMove.hFence ? "horizontal" : "vertical")
                << " fence at (" << selectedMove.pos.first << ", "
                << selectedMove.pos.second << ")";
    }
    std::cout << " (took " << elapsed.count() << " seconds)" << std::endl;

    // Apply the move
    currentState = *currentState.applyMove(selectedMove);
    currentState.displayBoard();

    // Check for winner
    int winner = checkWinner(currentState);
    if (winner != -1) {
      std::cout << "Player " << winner << " wins!" << std::endl;
      break;
    }

    turn++;
  }

  if (!isTerminal(currentState)) {
    std::cout << "Game ended after " << turn << " turns without a winner."
              << std::endl;
  }

  return 0;
}