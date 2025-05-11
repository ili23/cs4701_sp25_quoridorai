#include "game_state_tuple.hpp"

#include <iostream>
#include <queue>

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

Move::Move() {}

Gamestate::Gamestate() {
  p1Pos.first = 0;
  p1Pos.second = 4;
  p2Pos.first = 8;
  p2Pos.second = 4;

  p1Turn = true;
}

void Gamestate::displayBoard() {
  std::cout << "Current board state:" << std::endl;
  std::cout << "Player " << (p1Turn ? "0" : "1") << " to move" << std::endl;
  std::cout << "Player 0 fences left: " << p1Fences << std::endl;
  std::cout << "Player 1 fences left: " << p2Fences << std::endl;
  
  // Pre-compute which grid points have fences passing through them (for 2-unit length display)
  bool hFenceExtended[kBoardSize][kBoardSize] = {};
  bool vFenceExtended[kBoardSize][kBoardSize] = {};
  
  // Mark horizontal fences (each fence is 2 units long)
  for (int row = 0; row < kBoardSize - 1; row++) {
    for (int col = 0; col < kBoardSize - 1; col++) {
      if (hFences[row][col]) {
        hFenceExtended[row][col] = true;
        // Extend fence horizontally if possible (for 2-unit length)
        if (col < kBoardSize - 2) {
          hFenceExtended[row][col+1] = true;
        }
      }
    }
  }
  
  // Mark vertical fences (each fence is 2 units long)
  for (int row = 0; row < kBoardSize - 1; row++) {
    for (int col = 0; col < kBoardSize - 1; col++) {
      if (vFences[row][col]) {
        vFenceExtended[row][col] = true;
        // Extend fence vertically if possible (for 2-unit length)
        if (row < kBoardSize - 2) {
          vFenceExtended[row+1][col] = true;
        }
      }
    }
  }

  // Print board without coordinate numbers
  for (int row = 0; row < kBoardSize; row++) {
    // Print cells and vertical fences
    std::cout << "  "; // Indent
    for (int col = 0; col < kBoardSize; col++) {
      // Print the cell content
      if (std::make_pair(row, col) == p1Pos) {
        std::cout << " 0 ";
      } else if (std::make_pair(row, col) == p2Pos) {
        std::cout << " 1 ";
      } else {
        std::cout << " . ";
      }
      
      // Print vertical fence (if not at right edge)
      if (col < kBoardSize - 1) {
        if (vFenceExtended[row][col]) {
          std::cout << "║";
        } else {
          std::cout << "│";
        }
      }
    }
    std::cout << std::endl;
    
    // Print horizontal fences (if not at bottom edge)
    if (row < kBoardSize - 1) {
      std::cout << "  "; // Indent
      for (int col = 0; col < kBoardSize; col++) {
        if (hFenceExtended[row][col]) {
          std::cout << "═══";
        } else {
          std::cout << "───";
        }
        
        // Print intersection if not at right edge
        if (col < kBoardSize - 1) {
          // Check for possible fence crossings (which should never happen in a valid game)
          bool hasVertical = vFenceExtended[row][col];
          bool hasHorizontal = hFenceExtended[row][col];
          
          if (hasVertical && hasHorizontal) {
            std::cout << "╬"; // Both fences (shouldn't happen in valid game)
          } else if (hasVertical) {
            std::cout << "╫"; // Vertical fence only
          } else if (hasHorizontal) {
            std::cout << "╪"; // Horizontal fence only
          } else {
            std::cout << "┼"; // No fence
          }
        }
      }
      std::cout << std::endl;
    }
  }
  
  // Print player positions and fence counts again for clarity
  std::cout << "Player 0 position: (" << p1Pos.first << "," << p1Pos.second << "), Fences: " << p1Fences << std::endl;
  std::cout << "Player 1 position: (" << p2Pos.first << "," << p2Pos.second << "), Fences: " << p2Fences << std::endl;
  std::cout << std::endl;
}

std::unique_ptr<Gamestate> Gamestate::applyMove(const Move &m) const {
  std::unique_ptr<Gamestate> g = std::make_unique<Gamestate>(*this);

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

  return g;
}

bool inBounds(std::pair<int, int> x) {
  return x.first >= 0 && x.first < kBoardSize && x.second >= 0 &&
         x.second < kBoardSize;
}

std::vector<Move> Gamestate::getMoves() {
  std::vector<Move> moves;

  std::pair<int, int> startingPoint = p1Turn ? p1Pos : p2Pos;
  std::pair<int, int> otherPawn = p1Turn ? p2Pos : p1Pos;

  // Basic directional moves (up, down, left, right)
  // Up - row decreases
  std::pair<int, int> target =
      std::make_pair(startingPoint.first - 1, startingPoint.second);

  // Check for horizontal fence between current row and row above
  if (inBounds(target) && target != otherPawn &&
      !hFences[startingPoint.first - 1][startingPoint.second]) {
    moves.emplace_back(target);
  }

  // Down - row increases 
  target = std::make_pair(startingPoint.first + 1, startingPoint.second);

  // Check for horizontal fence between current row and row below
  if (inBounds(target) && target != otherPawn &&
      !hFences[startingPoint.first][startingPoint.second]) {
    moves.emplace_back(target);
  }

  // Left - column decreases
  target = std::make_pair(startingPoint.first, startingPoint.second - 1);

  // Check for vertical fence between current column and column to the left
  if (inBounds(target) && target != otherPawn &&
      !vFences[startingPoint.first][startingPoint.second - 1]) {
    moves.emplace_back(target);
  }

  // Right - column increases
  target = std::make_pair(startingPoint.first, startingPoint.second + 1);

  // Check for vertical fence between current column and column to the right
  if (inBounds(target) && target != otherPawn &&
      !vFences[startingPoint.first][startingPoint.second]) {
    moves.emplace_back(target);
  }

  // Jump moves over opponent (matching Python implementation)
  // Check if opponent is adjacent
  int drow = otherPawn.first - startingPoint.first;
  int dcol = otherPawn.second - startingPoint.second;
  
  if (abs(drow) + abs(dcol) == 1) {  // Adjacent pawns
    // Direction from player to opponent
    // Try straight jump over opponent
    std::pair<int, int> jumpTarget = std::make_pair(
        otherPawn.first + drow, 
        otherPawn.second + dcol
    );
    
    // Check if jump target is valid
    if (inBounds(jumpTarget)) {
      bool canJump = false;
      
      // Check fence between opponent and jump target
      if (dcol == 0) {  // Vertical jump (north/south)
        if (drow > 0) {  // Jumping down (increasing row)
          canJump = !hFences[otherPawn.first][otherPawn.second];
        } else {  // Jumping up (decreasing row)
          canJump = !hFences[otherPawn.first - 1][otherPawn.second];
        }
      } else {  // Horizontal jump (east/west)
        if (dcol > 0) {  // Jumping right (increasing column)
          canJump = !vFences[otherPawn.first][otherPawn.second];
        } else {  // Jumping left (decreasing column)
          canJump = !vFences[otherPawn.first][otherPawn.second - 1];
        }
      }
      
      if (canJump) {
        moves.emplace_back(jumpTarget);
      } else {
        // If straight jump is blocked, try diagonal jumps
        std::vector<std::pair<int, int>> diagonals;
        
        // Add potential diagonal jumps (perpendicular to straight jump)
        if (dcol == 0) {  // If moving vertically, try horizontal diagonals
          diagonals.push_back(std::make_pair(otherPawn.first, otherPawn.second + 1)); // right
          diagonals.push_back(std::make_pair(otherPawn.first, otherPawn.second - 1)); // left
        } else {  // If moving horizontally, try vertical diagonals
          diagonals.push_back(std::make_pair(otherPawn.first - 1, otherPawn.second)); // up
          diagonals.push_back(std::make_pair(otherPawn.first + 1, otherPawn.second)); // down
        }
        
        // Check each diagonal jump
        for (const auto& diagTarget : diagonals) {
          if (inBounds(diagTarget) && diagTarget != startingPoint) {
            bool canDiagJump = false;
            
            // Check fence between opponent and diagonal target
            if (diagTarget.second == otherPawn.second + 1) {  // Jumping right
              canDiagJump = !vFences[otherPawn.first][otherPawn.second];
            } else if (diagTarget.second == otherPawn.second - 1) {  // Jumping left
              canDiagJump = !vFences[otherPawn.first][diagTarget.second];
            } else if (diagTarget.first == otherPawn.first - 1) {  // Jumping up
              canDiagJump = !hFences[diagTarget.first][otherPawn.second];
            } else if (diagTarget.first == otherPawn.first + 1) {  // Jumping down
              canDiagJump = !hFences[otherPawn.first][otherPawn.second];
            }
            
            if (canDiagJump) {
              moves.emplace_back(diagTarget);
            }
          }
        }
      }
    }
  }

  // Return if the current player has no more fences to place
  if ((p1Turn ? p1Fences : p2Fences) <= 0) {
    return moves;
  }

  // Find valid fence moves
  for (int row = 0; row < kBoardSize - 1; row++) {
    for (int col = 0; col < kBoardSize - 1; col++) {
      // Only consider valid fence placements where one more fence won't block paths
      if (!hFences[row][col]) {
        // Check if horizontal fence would cross with vertical fence
        bool wouldCross = false;
        
        // Check for crossing with vertical fence at this position or next position
        if (vFences[row][col] || (col < kBoardSize - 2 && vFences[row][col+1])) {
          wouldCross = true;
        }
        
        // Check for adjacent fence that would conflict with 2-unit length
        bool conflictLength = false;
        if ((row > 0 && hFences[row-1][col]) ||  // fence above
            (row < kBoardSize - 2 && hFences[row+1][col]) ||  // fence below
            (col > 0 && hFences[row][col-1]) ||  // fence to the left
            (col < kBoardSize - 2 && hFences[row][col+1])) {  // fence to the right
          conflictLength = true;
        }
        
        // For 2-unit long fences, make sure there's space for the full fence
        if (col >= kBoardSize - 2) {
          conflictLength = true;
        }
        
        if (!wouldCross && !conflictLength) {
          // Temporarily place fence to check path to end
          hFences[row][col] = true;
          
          if (pathToEnd(true) && pathToEnd(false)) {
            moves.emplace_back(true, std::make_pair(row, col));
          }
          
          // Remove temporary fence
          hFences[row][col] = false;
        }
      }
      
      if (!vFences[row][col]) {
        // Check if vertical fence would cross with horizontal fence
        bool wouldCross = false;
        
        // Check for crossing with horizontal fence at this position or next position
        if (hFences[row][col] || (row < kBoardSize - 2 && hFences[row+1][col])) {
          wouldCross = true;
        }
        
        // Check for adjacent fence that would conflict with 2-unit length
        bool conflictLength = false;
        if ((row > 0 && vFences[row-1][col]) ||  // fence above
            (row < kBoardSize - 2 && vFences[row+1][col]) ||  // fence below
            (col > 0 && vFences[row][col-1]) ||  // fence to the left
            (col < kBoardSize - 2 && vFences[row][col+1])) {  // fence to the right
          conflictLength = true;
        }
        
        // For 2-unit long fences, make sure there's space for the full fence
        if (row >= kBoardSize - 2) {
          conflictLength = true;
        }
        
        if (!wouldCross && !conflictLength) {
          // Temporarily place fence to check path to end
          vFences[row][col] = true;
          
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

void Gamestate::displayAllMoves() {
  std::vector<Move> allMoves = getMoves();

  for (Move m : allMoves) {
    auto v = applyMove(m);
    v->displayBoard();
  }
}

bool Gamestate::pathToEnd(bool p1) {
  // Early optimization: only do path check if enough fences have been placed
  // Similar to the Python implementation's optimization
  int totalFencesPlaced = 2 * kStartingFences - (p1Fences + p2Fences);
  if (totalFencesPlaced <= 4) {
    // If fewer than 5 fences have been placed, a path must exist
    return true;
  }

  bool reachable[kBoardSize][kBoardSize] = {};

  std::queue<std::pair<int, int>> toSearch;

  // Start at current player position
  toSearch.push(p1 ? p1Pos : p2Pos);
  reachable[toSearch.front().first][toSearch.front().second] = true;

  std::pair<int, int> otherPawn = p1 ? p2Pos : p1Pos;

  std::pair<int, int> target;
  std::pair<int, int> startingPoint;

  // Goal row for the player
  // In new coordinate system (row, col):
  // Player 1 (p1) needs to reach the bottom row (row = BOARD_SIZE - 1)
  // Player 2 (p2) needs to reach the top row (row = 0)
  int goalRow = p1 ? kBoardSize - 1 : 0;

  // BFS to find a path to the goal
  while (!toSearch.empty()) {
    startingPoint = toSearch.front();
    
    // Check if we've reached the goal row
    if (startingPoint.first == goalRow) {
      return true;
    }

    // Try moving up (decrease row)
    target = std::make_pair(startingPoint.first - 1, startingPoint.second);
    if (inBounds(target) && !reachable[target.first][target.second] &&
        target != otherPawn &&
        !hFences[startingPoint.first - 1][startingPoint.second]) {
      reachable[target.first][target.second] = true;
      toSearch.emplace(target);
    }

    // Try moving down (increase row)
    target = std::make_pair(startingPoint.first + 1, startingPoint.second);
    if (inBounds(target) && !reachable[target.first][target.second] &&
        target != otherPawn &&
        !hFences[startingPoint.first][startingPoint.second]) {
      reachable[target.first][target.second] = true;
      toSearch.emplace(target);
    }

    // Try moving left (decrease column)
    target = std::make_pair(startingPoint.first, startingPoint.second - 1);
    if (inBounds(target) && !reachable[target.first][target.second] &&
        target != otherPawn &&
        !vFences[startingPoint.first][startingPoint.second - 1]) {
      reachable[target.first][target.second] = true;
      toSearch.emplace(target);
    }

    // Try moving right (increase column)
    target = std::make_pair(startingPoint.first, startingPoint.second + 1);
    if (inBounds(target) && !reachable[target.first][target.second] &&
        target != otherPawn &&
        !vFences[startingPoint.first][startingPoint.second]) {
      reachable[target.first][target.second] = true;
      toSearch.emplace(target);
    }

    toSearch.pop();
  }

  // Check if any position in the goal row is reachable
  for (int col = 0; col < kBoardSize; col++) {
    if (reachable[goalRow][col]) return true;
  }

  return false;
}
