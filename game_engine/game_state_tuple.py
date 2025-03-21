from collections import deque
import copy

# Constants
PLAYER_ONE = 0
PLAYER_TWO = 1
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
BOARD_SIZE = 9

# Game state is represented as a tuple of:
# (
#   (player_0_pos, player_1_pos),  # Pawns positions
#   (player_0_fences, player_1_fences),  # Remaining fences for each player
#   tuple(tuple(row) for row in horizontal_fences),  # Horizontal fences (as tuples of booleans)
#   tuple(tuple(row) for row in vertical_fences),  # Vertical fences (as tuples of booleans)
#   current_player  # Current player (0 or 1)
# )

def create_initial_state():
    """Create the initial game state tuple"""
    # Initial pawn positions
    pawns = ((0, 4), (8, 4))
    
    # Initial fence counts
    fences = (10, 10)
    
    # Initial fence layouts (all False)
    h_fences = tuple(tuple(False for _ in range(BOARD_SIZE - 1)) for _ in range(BOARD_SIZE - 1))
    v_fences = tuple(tuple(False for _ in range(BOARD_SIZE - 1)) for _ in range(BOARD_SIZE - 1))
    
    # Initial player
    current_player = PLAYER_ONE
    
    return (pawns, fences, h_fences, v_fences, current_player)

def is_fence_between(state, pos1, pos2):
    """Check if there's a fence between two adjacent positions"""
    _, _, h_fences, v_fences, _ = state
    x1, y1 = pos1
    x2, y2 = pos2
    
    # Ensure positions are within valid range
    if not (0 <= x1 < BOARD_SIZE and 0 <= y1 < BOARD_SIZE and 
            0 <= x2 < BOARD_SIZE and 0 <= y2 < BOARD_SIZE):
        return True  # Out of bounds is treated as blocked
    
    if x1 == x2:  # Moving vertically
        min_y = min(y1, y2)
        # Check horizontal fence blocking vertical movement
        if x1 < BOARD_SIZE - 1 and min_y < BOARD_SIZE - 1:
            if h_fences[x1][min_y]:
                return True
    elif y1 == y2:  # Moving horizontally
        min_x = min(x1, x2)
        # Check vertical fence blocking horizontal movement
        if min_x < BOARD_SIZE - 1 and y1 < BOARD_SIZE - 1:
            if v_fences[min_x][y1]:
                return True
    return False

def get_valid_pawn_moves(state, player):
    """Get all valid pawn moves for a player"""
    pawns, _, h_fences, v_fences, _ = state
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
        # Check straight jump
        jump_x, jump_y = 2*opp_x - x, 2*opp_y - y
        if 0 <= jump_x < BOARD_SIZE and 0 <= jump_y < BOARD_SIZE:
            if not is_fence_between(state, (opp_x, opp_y), (jump_x, jump_y)):
                moves.append(("move", (jump_x, jump_y)))
        
        # Check diagonal jumps when straight jump is blocked
        if is_fence_between(state, (opp_x, opp_y), (2*opp_x - x, 2*opp_y - y)):
            for d_x, d_y in DIRECTIONS:
                diag_x, diag_y = opp_x + d_x, opp_y + d_y
                if (diag_x, diag_y) != (x, y) and 0 <= diag_x < BOARD_SIZE and 0 <= diag_y < BOARD_SIZE:
                    if not is_fence_between(state, (opp_x, opp_y), (diag_x, diag_y)):
                        moves.append(("move", (diag_x, diag_y)))
    
    return moves

def path_exists(state, player, goal_row):
    """BFS to check if a path exists from player's position to goal row"""
    pawns, _, _, _, _ = state
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
    x, y = position
    pawns, _, h_fences, v_fences, _ = state
    
    # Check boundary conditions
    if not (0 <= x < BOARD_SIZE - 1 and 0 <= y < BOARD_SIZE - 1):
        return False
    
    # Check if fence already exists
    if orientation == 'H':
        if h_fences[x][y]:
            return False
        # Check for crossing fences
        if y > 0 and y < BOARD_SIZE - 2:
            if v_fences[x][y-1] and v_fences[x][y]:
                return False
    elif orientation == 'V':
        if v_fences[x][y]:
            return False
        # Check for crossing fences
        if x > 0 and x < BOARD_SIZE - 2:
            if h_fences[x-1][y] and h_fences[x][y]:
                return False
    
    # Place fence temporarily to check if it blocks all paths
    # We need to create mutable copies of our fence tuples
    h_fences_list = [list(row) for row in h_fences]
    v_fences_list = [list(row) for row in v_fences]
    
    if orientation == 'H':
        h_fences_list[x][y] = True
    else:
        v_fences_list[x][y] = True
    
    # Convert back to tuples for our functions
    temp_h_fences = tuple(tuple(row) for row in h_fences_list)
    temp_v_fences = tuple(tuple(row) for row in v_fences_list)
    
    # Create a temporary state with the fence placed
    temp_state = (pawns, (0, 0), temp_h_fences, temp_v_fences, 0)  # Fence counts don't matter here
    
    # Check if both players still have paths to win
    return path_exists_for_both_players(temp_state)

def get_valid_fence_moves(state, player):
    """Get all valid fence placements for a player"""
    _, fences, _, _, _ = state
    
    # If player has no fences left, return empty list
    if fences[player] <= 0:
        return []
    
    fence_moves = []
    # Check all possible fence placements
    for i in range(BOARD_SIZE - 1):
        for j in range(BOARD_SIZE - 1):
            if is_valid_fence_placement(state, (i, j), 'H'):
                fence_moves.append(("fence", (i, j), "H"))
            if is_valid_fence_placement(state, (i, j), 'V'):
                fence_moves.append(("fence", (i, j), "V"))
    
    return fence_moves

def get_possible_moves(state):
    """Get all possible moves for the current player"""
    pawns, fences, _, _, current_player = state
    
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
    pawns, fences, h_fences, v_fences, current_player = state
    pawns = list(pawns)
    fences = list(fences)
    
    if move[0] == "move":
        # Handle pawn move
        _, new_position = move
        pawns[current_player] = new_position
        
        # Convert lists back to tuples
        pawns = tuple(pawns)
        fences = tuple(fences)
        
        # Check if this results in a win
        winner = check_winner(state)
        if winner is not None:
            # If there's a winner, we still return the state
            # The caller should check for a winner after applying a move
            return (pawns, fences, h_fences, v_fences, 1 - current_player)
        
        # Switch player
        return (pawns, fences, h_fences, v_fences, 1 - current_player)
    
    elif move[0] == "fence":
        # Handle fence placement
        _, position, orientation = move
        x, y = position
        
        # Convert fence tuples to lists so we can modify them
        h_fences_list = [list(row) for row in h_fences]
        v_fences_list = [list(row) for row in v_fences]
        
        if orientation == 'H':
            h_fences_list[x][y] = True
            h_fences = tuple(tuple(row) for row in h_fences_list)
        else:  # orientation == 'V'
            v_fences_list[x][y] = True
            v_fences = tuple(tuple(row) for row in v_fences_list)
        
        # Update fence count
        fences[current_player] -= 1
        
        # Convert lists back to tuples
        pawns = tuple(pawns)
        fences = tuple(fences)
        
        # Switch player
        return (pawns, fences, h_fences, v_fences, 1 - current_player)

def check_winner(state):
    """Check if either player has won the game"""
    pawns, _, _, _, _ = state
    x0, y0 = pawns[0]
    x1, y1 = pawns[1]
    
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
    pawns, fences, h_fences, v_fences, current_player = state
    
    # ANSI color codes for better visualization
    BLUE = '\033[94m'    # For placed walls
    GRAY = '\033[90m'    # For potential wall slots
    GREEN = '\033[92m'   # For winner
    RESET = '\033[0m'    # Reset color
    
    print("Current board state:\n")
    winner = check_winner(state)
    if winner is not None:
        print(f"{GREEN}Player {winner} has won the game!{RESET}")
    
    for row in range(BOARD_SIZE):
        # Print horizontal fences and pawns
        for col in range(BOARD_SIZE):
            if (row, col) == pawns[0]:
                print(" 0 ", end="")
            elif (row, col) == pawns[1]:
                print(" 1 ", end="")
            else:
                print(" . ", end="")

            if col < BOARD_SIZE - 1:
                if row == BOARD_SIZE - 1:
                    has_fence = col > 0 and row > 0 and v_fences[col][row-1]
                elif row < BOARD_SIZE - 1:
                    has_fence = (col > 0 and row > 0 and v_fences[col-1][row-1]) or (col < BOARD_SIZE - 1 and v_fences[col][row])
                else:
                    has_fence = False
                    
                if has_fence:
                    print(f"{BLUE}║{RESET}", end="")
                else:
                    print(f"{GRAY}│{RESET}", end="")

        print()

        if row < BOARD_SIZE - 1:
            for col in range(BOARD_SIZE):
                if col == BOARD_SIZE - 1:
                    has_fence = row > 0 and col > 0 and h_fences[row][col-1]
                elif col < BOARD_SIZE - 1:
                    has_fence = h_fences[row][col] or (row > 0 and col > 0 and h_fences[row][col-1])
                else:
                    has_fence = False
                    
                if has_fence:
                    print(f"{BLUE}═══{RESET}", end="")
                else:
                    print(f"{GRAY}───{RESET}", end="")
                if col < BOARD_SIZE - 1:
                    print(f"{GRAY}╬{RESET}", end="")  # Intersection for a cleaner grid
            print()
    
    print(f"Current player: {current_player}")
    print(f"Player 0 fences left: {fences[0]}")
    print(f"Player 1 fences left: {fences[1]}")
    
    if winner is not None:
        print(f"{GREEN}Player {winner} has won the game!{RESET}")
    print()

# Example usage
def example_gameplay():
    # Create initial state
    state = create_initial_state()
    print_game_state(state)
    
    # Apply a sequence of moves
    moves = [
        ("move", (1, 4)),
        ("move", (7, 4)),
        ("move", (2, 4)),
        ("move", (6, 4)), 
        ("move", (3, 4)),
        ("move", (5, 4)),
        ("move", (4, 4)),
        ("fence", (0, 1), "V")
    ]
    
    for move in moves:
        print(f"Applying move: {move}")
        state = apply_move(state, move)
        print_game_state(state)
        
        # Check for winner after each move
        winner = check_winner(state)
        if winner is not None:
            print(f"Player {winner} wins!")
            break
    
    # Show available moves for current player
    current_moves = get_possible_moves(state)
    print(f"Available moves for player {state[4]}: {current_moves[:5]}...")
    print(f"Total available moves: {len(current_moves)}")

if __name__ == "__main__":
    example_gameplay() 