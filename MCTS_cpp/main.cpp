#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

#include "MCTS.hpp"

// Function to handle human player input
Move getHumanMove(Gamestate& state) {
  std::vector<Move> possibleMoves = state.getMoves();
  
  // Separate pawn moves and fence moves
  std::vector<Move> pawnMoves;
  std::vector<Move> hFenceMoves;
  std::vector<Move> vFenceMoves;
  
  for (const auto& move : possibleMoves) {
    if (move.pawnMove) {
      pawnMoves.push_back(move);
    } else if (move.hFence) {
      hFenceMoves.push_back(move);
    } else {
      vFenceMoves.push_back(move);
    }
  }
  
  // Display move options
  std::cout << "\nPawn moves:" << std::endl;
  for (size_t i = 0; i < pawnMoves.size(); i++) {
    std::cout << i + 1 << ". Move to (" << pawnMoves[i].pos.first << ", " 
              << pawnMoves[i].pos.second << ")" << std::endl;
  }
  
  std::cout << "\nFence moves:" << std::endl;
  std::cout << "Horizontal fence moves: " << hFenceMoves.size() << std::endl;
  std::cout << "Vertical fence moves: " << vFenceMoves.size() << std::endl;
  
  // Get user input
  bool validMove = false;
  Move chosenMove;
  
  while (!validMove) {
    std::cout << "\nEnter 'p' for pawn move or 'f' for fence placement: ";
    std::string moveType;
    std::cin >> moveType;
    
    if (moveType == "p") {
      if (pawnMoves.empty()) {
        std::cout << "No valid pawn moves available!" << std::endl;
        continue;
      }
      
      std::cout << "Enter pawn move number (1-" << pawnMoves.size() << "): ";
      int moveIndex;
      if (std::cin >> moveIndex && moveIndex >= 1 && moveIndex <= static_cast<int>(pawnMoves.size())) {
        chosenMove = pawnMoves[moveIndex - 1];
        validMove = true;
      } else {
        std::cout << "Invalid move index! Try again." << std::endl;
        std::cin.clear();
        std::cin.ignore(10000, '\n');
      }
    } else if (moveType == "f") {
      if (hFenceMoves.empty() && vFenceMoves.empty()) {
        std::cout << "No valid fence moves available!" << std::endl;
        continue;
      }
      
      std::cout << "Enter fence orientation (h/v): ";
      std::string orientation;
      std::cin >> orientation;
      
      if (orientation == "h") {
        if (hFenceMoves.empty()) {
          std::cout << "No valid horizontal fence moves available!" << std::endl;
          continue;
        }
        
        std::cout << "Enter coordinates as 'row,col' (e.g., '3,4'): ";
        std::string coords;
        std::cin >> coords;
        
        int row, col;
        char comma;
        std::stringstream ss(coords);
        if (ss >> row >> comma >> col && comma == ',') {
          // Find the matching fence move
          auto it = std::find_if(hFenceMoves.begin(), hFenceMoves.end(),
                              [row, col](const Move& m) {
                                return m.pos.first == row && m.pos.second == col;
                              });
          
          if (it != hFenceMoves.end()) {
            chosenMove = *it;
            validMove = true;
          } else {
            std::cout << "Invalid fence placement! Try again." << std::endl;
          }
        } else {
          std::cout << "Invalid coordinate format! Use 'row,col'." << std::endl;
          std::cin.clear();
          std::cin.ignore(10000, '\n');
        }
      } else if (orientation == "v") {
        if (vFenceMoves.empty()) {
          std::cout << "No valid vertical fence moves available!" << std::endl;
          continue;
        }
        
        std::cout << "Enter coordinates as 'row,col' (e.g., '3,4'): ";
        std::string coords;
        std::cin >> coords;
        
        int row, col;
        char comma;
        std::stringstream ss(coords);
        if (ss >> row >> comma >> col && comma == ',') {
          // Find the matching fence move
          auto it = std::find_if(vFenceMoves.begin(), vFenceMoves.end(),
                              [row, col](const Move& m) {
                                return m.pos.first == row && m.pos.second == col;
                              });
          
          if (it != vFenceMoves.end()) {
            chosenMove = *it;
            validMove = true;
          } else {
            std::cout << "Invalid fence placement! Try again." << std::endl;
          }
        } else {
          std::cout << "Invalid coordinate format! Use 'row,col'." << std::endl;
          std::cin.clear();
          std::cin.ignore(10000, '\n');
        }
      } else {
        std::cout << "Invalid orientation! Use 'h' or 'v'." << std::endl;
      }
    } else {
      std::cout << "Invalid input! Enter 'p' for pawn or 'f' for fence." << std::endl;
    }
  }
  
  return chosenMove;
}

// Main game loop function
void gameLoop(bool player0IsBot, bool player1IsBot) {
  // Create an initial gamestate
  Gamestate initialState;
  std::cout << "Initial board state:" << std::endl;
  initialState.displayBoard();
  
  // Create MCTS agents
  Agent agent0(1000);  // Increased from 50 to 500 iterations
  Agent agent1(1000);  // Increased from 50 to 500 iterations
  
  // Game loop
  Gamestate currentState = initialState;
  int moveCount = 0;
  
  while (!isTerminal(currentState)) {
    std::cout << "\nMove " << moveCount + 1 << ":" << std::endl;
    bool currentPlayerIsBot = currentState.p1Turn ? player0IsBot : player1IsBot;
    int currentPlayer = currentState.p1Turn ? 0 : 1;
    
    std::cout << "Current player: " << currentPlayer << std::endl;
    std::cout << "Player 0 position: (" << currentState.p1Pos.first << "," << currentState.p1Pos.second << ")" << std::endl;
    std::cout << "Player 1 position: (" << currentState.p2Pos.first << "," << currentState.p2Pos.second << ")" << std::endl;
    
    Move selectedMove;
    
    if (currentPlayerIsBot) {
      // Bot player - use MCTS
      std::cout << "Player " << currentPlayer << " (BOT) is thinking..." << std::endl;
      
      // Measure time for move selection
      auto start = std::chrono::high_resolution_clock::now();
      
      // Select a move using MCTS
      if (currentPlayer == 0) {
        selectedMove = agent0.selectMove(currentState);
      } else {
        selectedMove = agent1.selectMove(currentState);
      }
      
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      
      std::cout << "Player " << currentPlayer << " (BOT) took " << elapsed.count() << " seconds to decide." << std::endl;
    } else {
      // Human player - get input
      std::cout << "Player " << currentPlayer << " (HUMAN)'s turn" << std::endl;
      selectedMove = getHumanMove(currentState);
    }
    
    // Report the move
    std::cout << "Player " << currentPlayer;
    if (selectedMove.pawnMove) {
      std::cout << " moves pawn to (" << selectedMove.pos.first << ", "
                << selectedMove.pos.second << ")";
    } else {
      std::cout << " places "
                << (selectedMove.hFence ? "horizontal" : "vertical")
                << " fence at (" << selectedMove.pos.first << ", "
                << selectedMove.pos.second << ")";
    }
    std::cout << std::endl;
    
    // Apply the move
    currentState = *currentState.applyMove(selectedMove);
    currentState.displayBoard();
    
    // Check for winner
    int winner = checkWinner(currentState);
    if (winner != -1) {
      std::cout << "Player " << winner << " wins!" << std::endl;
      break;
    }
    
    moveCount++;
  }
  
  if (!isTerminal(currentState)) {
    std::cout << "Game ended after " << moveCount << " moves without a winner."
              << std::endl;
  }
}

int main() {
  std::cout << "Welcome to Quoridor!" << std::endl;
  std::cout << "Choose player types:" << std::endl;
  
  bool player0IsBot, player1IsBot;
  std::string input;
  
  std::cout << "Is Player 0 a bot? (y/n): ";
  std::cin >> input;
  player0IsBot = (input == "y" || input == "Y");
  
  std::cout << "Is Player 1 a bot? (y/n): ";
  std::cin >> input;
  player1IsBot = (input == "y" || input == "Y");
  
  try {
    gameLoop(player0IsBot, player1IsBot);
    std::cout << "Thanks for playing!" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
  }
  
  return 0;
}