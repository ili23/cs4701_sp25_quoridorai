#include "MCTS.hpp"
#include <iostream>
#include <chrono>

int main() {
    // Create an initial gamestate
    Gamestate initialState;
    std::cout << "Initial board state:" << std::endl;
    initialState.displayBoard();
    
    // Create MCTS agent
    Agent agent(1000); // 1000 iterations
    
    // Game loop
    Gamestate currentState = initialState;
    int turn = 0;
    
    while (!isTerminal(currentState) && turn < 30) {
        std::cout << "Turn " << turn + 1 << ":" << std::endl;
        
        // Measure time for move selection
        auto start = std::chrono::high_resolution_clock::now();
        
        // Select a move using MCTS
        Move selectedMove = agent.selectMove(currentState);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        // Apply the move
        std::cout << "Player " << (currentState.p1Turn ? "1" : "2");
        if (selectedMove.pawnMove) {
            std::cout << " moves pawn to (" << selectedMove.pos.first << ", " 
                     << selectedMove.pos.second << ")";
        } else {
            std::cout << " places " << (selectedMove.hFence ? "horizontal" : "vertical") 
                     << " fence at (" << selectedMove.pos.first << ", " << selectedMove.pos.second << ")";
        }
        std::cout << " (took " << elapsed.count() << " seconds)" << std::endl;
        
        // Apply the move
        currentState = *currentState.applyMove(selectedMove);
        currentState.displayBoard();
        
        // Check for winner
        int winner = checkWinner(currentState);
        if (winner != -1) {
            std::cout << "Player " << winner + 1 << " wins!" << std::endl;
            break;
        }
        
        turn++;
    }
    
    if (!isTerminal(currentState)) {
        std::cout << "Game ended after " << turn << " turns without a winner." << std::endl;
    }
    
    return 0;
} 