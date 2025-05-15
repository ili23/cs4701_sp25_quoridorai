from collections import deque
import copy

# Constants
PLAYER_ONE = 0
PLAYER_TWO = 1
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
BOARD_SIZE = 5

# Game state is represented as a tuple of:
# (
#   (player_0_pos, player_1_pos),  # Pawns positions
#   (player_0_fences, player_1_fences),  # Remaining fences for each player
#   tuple(tuple(row) for row in horizontal_fences),  # Horizontal fences (as tuples of booleans)
#   tuple(tuple(row) for row in vertical_fences),  # Vertical fences (as tuples of booleans)
#   current_player,  # Current player (0 or 1)
#   move_count  # Number of moves played so far
# )
def get_current_player(state):
    return state[4]

def create_initial_state():
    """Create the initial game state tuple"""
    # Initial pawn positions
    pawns = ((0, BOARD_SIZE // 2), (BOARD_SIZE - 1, BOARD_SIZE // 2))
    
    # Initial fence counts
    fences = (10, 10)
    
    # Initial fence layouts (all False)
    h_fences = tuple(tuple(False for _ in range(BOARD_SIZE - 1)) for _ in range(BOARD_SIZE - 1))
    v_fences = tuple(tuple(False for _ in range(BOARD_SIZE - 1)) for _ in range(BOARD_SIZE - 1))
    
    # Initial player
    current_player = PLAYER_ONE
    
    # Initial move count
    move_count = 0
    
    return (pawns, fences, h_fences, v_fences, current_player, move_count)

def is_fence_between(state, pos1, pos2):
    """Check if there's a fence between two adjacent positions"""
    _, _, h_fences, v_fences, _, _ = state
    row1, col1 = pos1
    row2, col2 = pos2
    
    # Ensure positions are within valid range
    if not (0 <= row1 < BOARD_SIZE and 0 <= col1 < BOARD_SIZE and 
            0 <= row2 < BOARD_SIZE and 0 <= col2 < BOARD_SIZE):
        return True  # Out of bounds is treated as blocked
    
    # Ensure positions are adjacent
    if abs(row1 - row2) + abs(col1 - col2) != 1:
        return True  # Non-adjacent positions are treated as blocked
    
    if row1 == row2:  # Moving horizontally
        min_col = min(col1, col2)
        # Check for vertical fence blocking horizontal movement
        row = row1
        
        # Check if there's a vertical fence at the column boundary
        if min_col < BOARD_SIZE - 1 and row < BOARD_SIZE - 1:
            if v_fences[row][min_col]:
                return True
                
        # Also check if there's a vertical fence one row up (since fences extend vertically)
        if min_col < BOARD_SIZE - 1 and row > 0:
            if row - 1 < BOARD_SIZE - 1 and v_fences[row-1][min_col]:
                return True
                
    elif col1 == col2:  # Moving vertically
        min_row = min(row1, row2)
        # Check for horizontal fence blocking vertical movement
        col = col1
        
        # Check if there's a horizontal fence at the row boundary
        if min_row < BOARD_SIZE - 1 and col < BOARD_SIZE - 1:
            if h_fences[min_row][col]:
                return True
        
        # Also check if there's a horizontal fence one column to the left (since fences extend horizontally)
        if min_row < BOARD_SIZE - 1 and col > 0:
            if col - 1 < BOARD_SIZE - 1 and h_fences[min_row][col-1]:
                return True
                
    return False

def get_valid_pawn_moves(state, player):
    """Get all valid pawn moves for a player"""
    pawns, _, h_fences, v_fences, _, _ = state
    moves = []
    x, y = pawns[player]
    opponent = 1 - player
    
    # Regular moves in four directions
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if (nx, ny) != pawns[opponent] and not is_fence_between(state, (x, y), (nx, ny)):
                moves.append(("move", (nx, ny)))
    
    # Jump moves over opponent
    opp_x, opp_y = pawns[opponent]
    if abs(x - opp_x) + abs(y - opp_y) == 1:  # Adjacent pawns
        # Direction from player to opponent
        dir_x = opp_x - x
        dir_y = opp_y - y
        
        # Calculate the position for a straight jump
        jump_x, jump_y = opp_x + dir_x, opp_y + dir_y
        
        # Check if straight jump is valid
        straight_jump_valid = (0 <= jump_x < BOARD_SIZE and 
                              0 <= jump_y < BOARD_SIZE and 
                              not is_fence_between(state, (opp_x, opp_y), (jump_x, jump_y)))
        
        if straight_jump_valid:
            moves.append(("move", (jump_x, jump_y)))
        else:
            # Straight jump is blocked or out of bounds, check diagonal jumps
            for d_x, d_y in DIRECTIONS:
                # Only consider actual diagonal jumps (perpendicular to straight jump)
                if (d_x != dir_x or d_y != dir_y) and (d_x != -dir_x or d_y != -dir_y):
                    diag_x, diag_y = opp_x + d_x, opp_y + d_y
                    # Check if diagonal position is valid and not occupied by the player
                    if ((diag_x, diag_y) != (x, y) and 
                        0 <= diag_x < BOARD_SIZE and 
                        0 <= diag_y < BOARD_SIZE):
                        # Check if there's no fence blocking the diagonal move
                        if not is_fence_between(state, (opp_x, opp_y), (diag_x, diag_y)):
                            moves.append(("move", (diag_x, diag_y)))
    
    return moves

def path_exists(state, player, goal_row):
    """BFS to check if a path exists from player's position to goal row"""
    pawns, remaining_fences, h_fences, v_fences, _, _ = state
    
    # Early optimization: only do this check for initial game state
    total_fences_placed = 20 - (remaining_fences[0] + remaining_fences[1])
    if total_fences_placed <= 4:
        # If fewer than 5 fences have been placed, a path must exist
        return True
        
    queue = deque([pawns[player]])
    visited = set([pawns[player]])
    
    while queue:
        x, y = queue.popleft()
        
        # Check if reached goal row
        if x == goal_row:
            return True
            
        # Try all four directions
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and (nx, ny) not in visited:
                # Check if move is valid (no fence blocking)
                if not is_fence_between(state, (x, y), (nx, ny)):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    
    return False

def path_exists_for_both_players(state):
    """Check if both players have paths to their goals"""
    # Check path exists for player 1 (top to bottom)
    if not path_exists(state, 0, BOARD_SIZE - 1):
        return False
        
    # Check path exists for player 2 (bottom to top)
    if not path_exists(state, 1, 0):
        return False
        
    return True

def is_valid_fence_placement(state, position, orientation):
    """Check if a fence placement is valid"""
    row, col = position
    pawns, _, h_fences, v_fences, _, _ = state
    
    # Check boundary conditions
    if not (0 <= row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1):
        return False
    
    # For 2-unit long fences, make sure there's space for the full fence
    if orientation == 'H' and col >= BOARD_SIZE - 1:
        return False
    if orientation == 'V' and row >= BOARD_SIZE - 1:
        return False
    
    # Check if fence already exists at the placement position
    if orientation == 'H':
        # Check if horizontal fence already exists at position
        if h_fences[row][col]:
            return False
            
        # Check for adjacent fence that would conflict with 2-unit length
        if col < BOARD_SIZE - 2 and h_fences[row][col+1]:
            return False
        
        if h_fences[row][col-1]:
            return False # fence immediately to the left extends to the right

        # Check for crossing with vertical fence
        # A horizontal fence at (row,col) crosses with a vertical fence at (row,col)
        if v_fences[row][col]:
            return False
        # Also check crossing at the end of the 2-unit fence
        if col < BOARD_SIZE - 2 and v_fences[row][col+1]:
            return False


         
    elif orientation == 'V':
        # Check if vertical fence already exists at position
        if v_fences[row][col]:
            return False
            
        # Check for adjacent fence that would conflict with 2-unit length
        if row < BOARD_SIZE - 2 and v_fences[row+1][col]:
            return False

        if v_fences[row-1][col]:
            return False # fence immediately above extends downwards
            
        # Check for crossing with horizontal fence
        # A vertical fence at (row,col) crosses with a horizontal fence at (row,col)
        if h_fences[row][col]:
            return False
        # Also check crossing at the end of the 2-unit fence
        if row < BOARD_SIZE - 2 and h_fences[row+1][col]:
            return False
    
    # Place fence temporarily to check if it blocks all paths
    # We need to create mutable copies of our fence tuples
    h_fences_list = [list(row) for row in h_fences]
    v_fences_list = [list(row) for row in v_fences]
    
    if orientation == 'H':
        h_fences_list[row][col] = True
    else:  # orientation == 'V'
        v_fences_list[row][col] = True
    
    # Convert back to tuples for our functions
    temp_h_fences = tuple(tuple(row) for row in h_fences_list)
    temp_v_fences = tuple(tuple(row) for row in v_fences_list)
    
    # Create a temporary state with the fence placed
    temp_state = (pawns, (0, 0), temp_h_fences, temp_v_fences, 0, 0)  # Fence counts don't matter here
    
    # Check if both players still have paths to win
    return path_exists_for_both_players(temp_state)

def get_valid_fence_moves(state, player):
    """Get all valid fence placements for a player"""
    _, fences, _, _, _, _ = state
    
    # If player has no fences left, return empty list
    if fences[player] <= 0:
        return []
    
    fence_moves = []
    # Check all possible fence placements
    for row in range(BOARD_SIZE - 1):
        for col in range(BOARD_SIZE - 1):
            if is_valid_fence_placement(state, (row, col), 'H'):
                fence_moves.append(("fence", (row, col), "H"))
            if is_valid_fence_placement(state, (row, col), 'V'):
                fence_moves.append(("fence", (row, col), "V"))
    
    return fence_moves

def get_possible_moves(state):
    """Get all possible moves for the current player"""
    pawns, fences, _, _, current_player, _ = state
    
    # Get pawn moves
    pawn_moves = get_valid_pawn_moves(state, current_player)
    
    # Get fence moves (if player has fences left)
    if fences[current_player] > 0:
        fence_moves = get_valid_fence_moves(state, current_player)
        return pawn_moves + fence_moves
    else:
        return pawn_moves

def apply_move(state, move):
    """Apply a move and return the new state. 
    Does not modify the original state tuple."""
    pawns, fences, h_fences, v_fences, current_player, move_count = state
    pawns = list(pawns)
    fences = list(fences)
    
    # Increment move count
    move_count += 1
    
    if move[0] == "move":
        # Handle pawn move
        _, new_position = move
        pawns[current_player] = new_position
        
        # Convert lists back to tuples
        pawns = tuple(pawns)
        fences = tuple(fences)
        
        # Create new state with updated pawn position
        new_state = (pawns, fences, h_fences, v_fences, 1 - current_player, move_count)
        
        # Check if this results in a win on the new state
        winner = check_winner(new_state)
        if winner is not None:
            # Return state with winner, caller should check for winner
            return new_state
        
        # Switch player and return
        return new_state
    
    elif move[0] == "fence":
        # Handle fence placement
        _, position, orientation = move
        row, col = position
        
        # Convert fence tuples to lists so we can modify them
        h_fences_list = [list(row) for row in h_fences]
        v_fences_list = [list(row) for row in v_fences]
        
        if orientation == 'H':
            # Place horizontal fence (store only the starting position)
            h_fences_list[row][col] = True
            h_fences = tuple(tuple(row) for row in h_fences_list)
        else:  # orientation == 'V'
            # Place vertical fence (store only the starting position)
            v_fences_list[row][col] = True
            v_fences = tuple(tuple(row) for row in v_fences_list)
        
        # Update fence count
        fences[current_player] -= 1
        
        # Convert lists back to tuples
        pawns = tuple(pawns)
        fences = tuple(fences)
        
        # Create new state and return
        new_state = (pawns, fences, h_fences, v_fences, 1 - current_player, move_count)
        return new_state

def check_winner(state):
    """Check if either player has won the game or if there's a tie"""
    pawns, _, _, _, _, move_count = state
    x0, y0 = pawns[0]
    x1, y1 = pawns[1]
    
    # Check for tie (move limit reached)
    if move_count >= 70:
        return 0.5  # Return 0.5 to indicate a tie
    
    # Player 0 wins by reaching the bottom row
    if x0 == BOARD_SIZE - 1:
        return 0
    
    # Player 1 wins by reaching the top row
    if x1 == 0:
        return 1
        
    # No winner yet
    return None

def print_game_state(state):
    """Print the current game state (board visualization)"""
    pawns, fences, h_fences, v_fences, current_player, move_count = state
    
    # ANSI color codes for better visualization
    BLUE = '\033[94m'    # For placed walls
    GRAY = '\033[90m'    # For potential wall slots
    GREEN = '\033[92m'   # For winner
    RESET = '\033[0m'    # Reset color
    
    print("Current board state:\n")
    winner = check_winner(state)
    if winner == 0.5:
        print(f"{GREEN}Game ended in a tie after {move_count} moves!{RESET}")
    elif winner is not None:
        print(f"{GREEN}Player {winner} has won the game!{RESET}")
    
    # Pre-compute which grid points have fences passing through them
    h_fence_grid = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    v_fence_grid = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # Mark horizontal fences (each fence is 2 units long)
    for row in range(BOARD_SIZE - 1):
        for col in range(BOARD_SIZE - 1):
            if h_fences[row][col]:
                h_fence_grid[row][col] = True
                if col < BOARD_SIZE - 1:  # Mark the extension horizontally (to next column)
                    h_fence_grid[row][col+1] = True
    
    # Mark vertical fences (each fence is 2 units long)
    for row in range(BOARD_SIZE - 1):
        for col in range(BOARD_SIZE - 1):
            if v_fences[row][col]:
                v_fence_grid[row][col] = True
                if row < BOARD_SIZE - 1:  # Mark the extension vertically (to next row)
                    v_fence_grid[row+1][col] = True
    # for row in range(BOARD_SIZE):
    #     print(h_fence_grid[row])
    # for row in range(BOARD_SIZE):
    #     print(v_fence_grid[row])
    # Print the board with pawns and fences
    for row in range(BOARD_SIZE):
        # Print pawns and vertical fences
        for col in range(BOARD_SIZE):
            # Print pawn or empty space
            if (row, col) == pawns[0]:
                print(" 0 ", end="")
            elif (row, col) == pawns[1]:
                print(" 1 ", end="")
            else:
                print(" . ", end="")
            
            # Print vertical fence if not at right edge
            if col < BOARD_SIZE - 1:
                has_fence = v_fence_grid[row][col]
                
                if has_fence:
                    print(f"{BLUE}║{RESET}", end="")
                else:
                    print(f"{GRAY}│{RESET}", end="")
        
        print()  # New line after row
        
        # Print horizontal fences if not at bottom edge
        if row < BOARD_SIZE - 1:
            for col in range(BOARD_SIZE):
                has_fence = h_fence_grid[row][col]
                
                if has_fence:
                    print(f"{BLUE}═══{RESET}", end="")
                else:
                    print(f"{GRAY}───{RESET}", end="")
                
                # Print intersection if not at right edge - always gray
                if col < BOARD_SIZE - 1:
                    print(f"{GRAY}╬{RESET}", end="")
            
            print()  # New line after horizontal fences
    
    print(f"Current player: {current_player}")
    print(f"Player 0 fences left: {fences[0]}")
    print(f"Player 1 fences left: {fences[1]}")
    print(f"Moves played: {move_count}")
    
    if winner == 0.5:
        print(f"{GREEN}Game ended in a tie after {move_count} moves!{RESET}")
    elif winner is not None:
        print(f"{GREEN}Player {winner} has won the game!{RESET}")
    print()

def is_terminal(state):
    return check_winner(state) is not None

def get_initial_state():
    """Alias for create_initial_state for better readability"""
    return create_initial_state()