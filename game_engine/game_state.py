from collections import deque
PLAYER_ONE = 0
PLAYER_TWO = 1
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
BOARD_SIZE = 9

class QuoridorGame:
    def __init__(self):
        self.board_size = BOARD_SIZE
        # Fix the data structure with unified player data
        self.pawns = [(0, 4), (8, 4)]
        self.fences = [10, 10]
        self.player_pawn_possible_moves = [[], []]
        self.fence_possible_moves = []  # Shared fence possible moves
        
        self.current_player = PLAYER_ONE
        self.horizontal_fences = [[False] * (self.board_size - 1) for _ in range(self.board_size - 1)]
        self.vertical_fences = [[False] * (self.board_size - 1) for _ in range(self.board_size - 1)]
        
        # Initialize possible pawn moves for both players
        self.update_pawn_possible_moves(PLAYER_ONE)
        self.update_pawn_possible_moves(PLAYER_TWO)
        # Initialize shared fence possible moves
        self.update_fence_possible_moves()
        self.winner = None
    
    def is_valid_pawn_move(self, player, new_position):
        """Check if a pawn move is valid"""
        x, y = self.pawns[player]
        nx, ny = new_position
        if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
            return False
        if (nx, ny) == self.pawns[1-player] or (nx, ny) == self.pawns[player]:
            return False
        # if self.is_fence_between((x, y), (nx, ny)):
        #     return False
        # if self.is_fence_between((nx, ny), (2*self.pawns[1-player][0] - x, 2*self.pawns[1-player][1] - y)):
        #     return False
        return True
    
    def update_pawn_possible_moves(self, player):
        """Update possible pawn moves for a player"""
        self.player_pawn_possible_moves[player] = []
        
        # Add possible pawn moves
        x, y = self.pawns[player]
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if self.is_valid_pawn_move(player, (nx, ny)):
                self.player_pawn_possible_moves[player].append(("move", (nx, ny)))
     
        # Add jump moves (over opponent)
        opp_x, opp_y = self.pawns[1 - player]
        if abs(x - opp_x) + abs(y - opp_y) == 1:  # Adjacent pawns
            # Check for jump straight over opponent
            jump_x, jump_y = 2*opp_x - x, 2*opp_y - y
            if 0 <= jump_x < self.board_size and 0 <= jump_y < self.board_size:
                # Check if there's no fence blocking the jump
                if not self.is_fence_between((opp_x, opp_y), (jump_x, jump_y)):
                    self.player_pawn_possible_moves[player].append(("move", (jump_x, jump_y)))
            
            # Check for diagonal jumps when straight jump is blocked
            if self.is_fence_between((opp_x, opp_y), (2*opp_x - x, 2*opp_y - y)):
                for d_x, d_y in DIRECTIONS:
                    diag_x, diag_y = opp_x + d_x, opp_y + d_y
                    if (diag_x, diag_y) != (x, y) and 0 <= diag_x < self.board_size and 0 <= diag_y < self.board_size:
                        if not self.is_fence_between((opp_x, opp_y), (diag_x, diag_y)):
                            self.player_pawn_possible_moves[player].append(("move", (diag_x, diag_y)))

    def update_fence_possible_moves(self):
        """Update possible fence placements (shared between players)"""
        self.fence_possible_moves = []
        
        # Add possible fence placements
        for i in range(self.board_size - 1):
            for j in range(self.board_size - 1):
                if self.is_valid_fence_placement((i, j), 'H'):
                    self.fence_possible_moves.append(("fence", (i, j), "H"))
                if self.is_valid_fence_placement((i, j), 'V'):
                    self.fence_possible_moves.append(("fence", (i, j), "V"))

    def update_possible_moves(self, player):
        """Update all possible moves for a player based on current game state"""
        self.update_pawn_possible_moves(player)
        # No need to update fence possible moves here, will be done separately

    def is_fence_between(self, pos1, pos2):
        """Check if there's a fence between two adjacent positions"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Ensure positions are within valid range for checking fences
        if not (0 <= x1 < self.board_size and 0 <= y1 < self.board_size and 
                0 <= x2 < self.board_size and 0 <= y2 < self.board_size):
            return True  # Out of bounds is treated as blocked
        
        if x1 == x2:  # Moving vertically
            min_y = min(y1, y2)
            # Check if there's a horizontal fence blocking the vertical movement
            if x1 < self.board_size - 1 and min_y < self.board_size - 1:
                if self.horizontal_fences[x1][min_y]:
                    return True
        elif y1 == y2:  # Moving horizontally
            min_x = min(x1, x2)
            # Check if there's a vertical fence blocking the horizontal movement
            if min_x < self.board_size - 1 and y1 < self.board_size - 1:
                if self.vertical_fences[min_x][y1]:
                    return True
        return False

    def get_possible_moves(self, player):
        """Get all possible moves for a player"""
        # If game is over or not this player's turn, return empty list
        if self.winner is not None or player != self.current_player:
            return []
        
        # Return pawn moves + fence moves (if player has fences left)
        if self.fences[player] > 0:
            return self.player_pawn_possible_moves[player] + self.fence_possible_moves
        else:
            return self.player_pawn_possible_moves[player]

    def step(self, move):
        # Check if game is already over
        if self.winner is not None:
            raise ValueError(f"Game is already won by Player {self.winner}")
        
        """Execute a move and update game state"""
        if len(move) == 2:  # Pawn move
            action, position = move
            if self.move_pawn(self.current_player, position):
                # Update pawn moves for both players
                self.update_pawn_possible_moves(PLAYER_ONE)
                self.update_pawn_possible_moves(PLAYER_TWO)
                # No need to update fence moves as pawn moves don't affect fence placements
        elif len(move) == 3:  # Fence placement
            action, position, orientation = move
            if self.place_fence(self.current_player, position, orientation):
                # Update pawn moves for both players (fence might affect valid moves)
                self.update_pawn_possible_moves(PLAYER_ONE)
                self.update_pawn_possible_moves(PLAYER_TWO)
                # Update shared fence possible moves
                self.update_fence_possible_moves()
        
        # Check if this move resulted in a win
        self.winner = self.check_winner()
        
        # If game is over, clear possible moves
        if self.winner is not None:
            self.player_pawn_possible_moves = [[], []]
            self.fence_possible_moves = []

    def move_pawn(self, player, new_position):
        # Precondition: new_position is in the player_pawn_possible_moves
        nx, ny = new_position            
        self.pawns[player] = new_position
        
        # Check if this move results in a win before switching players
        potential_winner = self.check_winner()
        if potential_winner is not None:
            self.winner = potential_winner
            return True
        
        self.current_player = 1 - self.current_player  # Switch player
        return True
        
    def place_fence(self, player, position, orientation):
        if self.fences[player] > 0 and self.is_valid_fence_placement(position, orientation):
            x, y = position
            if orientation == 'H':
                self.horizontal_fences[x][y] = True
            elif orientation == 'V':
                self.vertical_fences[x][y] = True
            self.fences[player] -= 1
            self.current_player = 1 - self.current_player  # Switch player
            
            # Update fence possible moves after placement
            self.update_fence_possible_moves()
            return True
        else:
            print("Invalid fence placement")
            return False

    def is_valid_fence_placement(self, pos, orientation):
        x, y = pos
        
        # Check boundary conditions
        if not (0 <= x < self.board_size and 0 <= y < self.board_size ):
            return False
        
        # Check if fence already exists
        if orientation == 'H':
            if self.horizontal_fences[x][y]:
                return False
            # Check for crossing fences (horizontal fence crossing vertical fence)
            if y > 0 and y < self.board_size - 1:
                if self.vertical_fences[x][y-1] and self.vertical_fences[x][y]:
                    return False
        elif orientation == 'V':
            if self.vertical_fences[x][y]:
                return False
            # Check for crossing fences (vertical fence crossing horizontal fence)
            if x > 0 and x < self.board_size - 1:
                if self.horizontal_fences[x-1][y] and self.horizontal_fences[x][y]:
                    return False
        
        # Place fence temporarily to check if it blocks all paths
        if orientation == 'H':
            self.horizontal_fences[x][y] = True
        else:
            self.vertical_fences[x][y] = True
            
        # Check if both players still have paths to win
        path_exists = self.path_exists_for_both_players()
        
        # Undo the temporary fence placement
        if orientation == 'H':
            self.horizontal_fences[x][y] = False
        else:
            self.vertical_fences[x][y] = False
            
        return path_exists

    def path_exists_for_both_players(self):
        # Check path exists for player 1 (top to bottom)
        if not self.path_exists(0, self.board_size - 1):
            return False
            
        # Check path exists for player 2 (bottom to top)
        if not self.path_exists(1, 0):
            return False
            
        return True
    
    def path_exists(self, player, goal_row):
        """BFS to check if a path exists from player's position to goal row"""
        queue = deque([self.pawns[player]])
        visited = set([self.pawns[player]])
        
        while queue:
            x, y = queue.popleft()
            
            # Check if reached goal row
            if x == goal_row:
                return True
                
            # Try all four directions
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and (nx, ny) not in visited:
                    # Check if move is valid (no fence blocking)
                    if not self.is_fence_between((x, y), (nx, ny)):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                        
        return False

    def print_game_state(self):
        # ANSI color codes
        BLUE = '\033[94m'    # For placed walls
        GRAY = '\033[90m'    # For potential wall slots
        GREEN = '\033[92m'   # For winner
        RESET = '\033[0m'    # Reset color
        
        print("Current board state:\n")
        if self.winner is not None:
            print(f"{GREEN}Player {self.winner} has won the game!{RESET}")
            return
        
        for row in range(self.board_size):
            # Print horizontal fences and pawns
            for col in range(self.board_size):
                if (row, col) == self.pawns[0]:
                    print(" 0 ", end="")
                elif (row, col) == self.pawns[1]:
                    print(" 1 ", end="")
                else:
                    print(" . ", end="")

                if col < self.board_size - 1:
                    if row == self.board_size - 1:
                        has_fence = col > 0 and self.vertical_fences[row-1][col]
                    elif row < self.board_size - 1:
                        has_fence = (col > 0 and self.vertical_fences[row-1][col]) or self.vertical_fences[row][col]
                    if has_fence:
                        print(f"{BLUE}║{RESET}", end="")
                    else:
                        print(f"{GRAY}│{RESET}", end="")

            print()

            if row < self.board_size - 1:
                for col in range(self.board_size):
                    if col == self.board_size - 1:
                        has_fence = row > 0 and self.horizontal_fences[row][col-1]
                    elif col < self.board_size - 1:
                        has_fence = self.horizontal_fences[row][col] or (row > 0 and self.horizontal_fences[row][col-1])
                    if has_fence:
                        print(f"{BLUE}═══{RESET}", end="")
                    else:
                        print(f"{GRAY}───{RESET}", end="")
                    if col < self.board_size - 1:
                        print(f"{GRAY}╬{RESET}", end="")  # Intersection for a cleaner grid
                print()
        print(f"Current player: {self.current_player}")
        print(f"Player 0 fences left: {self.fences[0]}")
        print(f"Player 1 fences left: {self.fences[1]}")
        
        if self.winner is not None:
            print(f"{GREEN}Player {self.winner} has won the game!{RESET}")
        else:
            print(f"Player 0 possible moves: {self.player_pawn_possible_moves[0]}")
            print(f"Player 1 possible moves: {self.player_pawn_possible_moves[1]}")
            # print(f"Possible fence moves: {self.fence_possible_moves}")
        print()

    def check_winner(self):
        """Check if either player has won the game"""
        x0, y0 = self.pawns[0]
        x1, y1 = self.pawns[1]
        
        # Player 0 wins by reaching the bottom row
        if x0 == self.board_size - 1:
            return 0
        
        # Player 1 wins by reaching the top row
        if x1 == 0:
            return 1
            
        # No winner yet
        return None


# Example usage
board = QuoridorGame()
board.print_game_state()
board.step(("move", (1, 4)))
board.step(("move", (7, 4)))
board.step(("move", (2, 4)))
board.step(("move", (6, 4)))
board.step(("move", (3, 4)))
board.step(("move", (5, 4)))
board.print_game_state()
board.step(("move", (4, 4)))
board.place_fence(0, (0, 1), 'V')
board.place_fence(1, (7, 1), 'V')
board.place_fence(0, (1, 1), 'H')
board.print_game_state()
print(board.vertical_fences)
# board.move_pawn(0, (1, 4))
# board.move_pawn(1, (7, 4))

# board.print_game_state()


