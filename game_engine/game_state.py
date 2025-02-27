from collections import deque

class QuoridorGame:
    def __init__(self):
        self.board_size = 9
        self.pawns = [(0, 4), (8, 4)]  # Player 1 and Player 2 starting positions
        self.fences = [10, 10]  # Each player starts with 10 fences
        self.current_player = 0
        self.horizontal_fences = [[False] * (self.board_size) for _ in range(self.board_size - 1)]
        self.vertical_fences = [[False] * (self.board_size - 1) for _ in range(self.board_size)]

    def move_pawn(self, player, new_position):
        # Validate player and position inputs
        if not isinstance(player, int) or player < 0:
            print("Player must be a positive integer")
            return
        if not isinstance(new_position, tuple) or len(new_position) != 2:
            print("New position must be a tuple of two coordinates")
            return
        nx, ny = new_position
        if not (isinstance(nx, int) and isinstance(ny, int)) or nx < 0 or ny < 0:
            print("Position coordinates must be positive integers")
            return
            
        if self.is_valid_pawn_move(player, new_position):
            self.pawns[player] = new_position
            self.current_player = 1 - self.current_player  # Switch player
        else:
            print("Invalid move")

    def is_valid_pawn_move(self, player, new_position):
        # Check if the move is within bounds and not blocked by a fence
        x, y = self.pawns[player]
        nx, ny = new_position
        if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
            return False
        if abs(nx - x) + abs(ny - y) != 1:
            return False
        # Check for fences
        if nx == x:
            if ny > y and self.horizontal_fences[x][y]:
                return False
            if ny < y and self.horizontal_fences[x][ny]:
                return False
        if ny == y:
            if nx > x and self.vertical_fences[x][y]:
                return False
            if nx < x and self.vertical_fences[nx][y]:
                return False
        return True

    def place_fence(self, player, position, orientation):
        if self.fences[player] > 0 and self.is_valid_fence_placement(player, position, orientation):
            x, y = position
            if orientation == 'H':
                self.horizontal_fences[x][y] = True
            elif orientation == 'V':
                self.vertical_fences[x][y] = True
            self.fences[player] -= 1
            self.current_player = 1 - self.current_player  # Switch player
        else:
            print("Invalid fence placement")

    def is_valid_fence_placement(self, player, pos, orientation):
        x, y = pos
        if orientation == 'H':
            if x > self.board_size - 1 or y > self.board_size - 1:
                return False
            if self.horizontal_fences[x][y]:
                return False
        elif orientation == 'V':
            if x > self.board_size - 1 or y > self.board_size:
                return False
            if self.vertical_fences[x][y]:
                return False
        # Ensure that the placement does not block all paths
        return True #self.path_exists_for_both_players()

    def path_exists_for_both_players(self):
        def bfs(start, goal_line):
            queue = deque([start])
            visited = set()
            while queue:
                x, y = queue.popleft()
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                if y == goal_line:
                    return True
                # Check all possible moves
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if self.is_valid_pawn_move(self.current_player, (nx, ny)):
                            queue.append((nx, ny))
            return False

        # Check path for both players
        return bfs(self.pawns[0], self.board_size - 1) and bfs(self.pawns[1], 0)

    def print_game_state(self):
        # ANSI color codes
        BLUE = '\033[94m'    # For placed walls
        GRAY = '\033[90m'    # For potential wall slots
        RESET = '\033[0m'    # Reset color
        
        print("Current board state:\n")
        
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
                    has_fence = self.vertical_fences[row][col] or (col > 0 and self.vertical_fences[row-1][col])
                    if has_fence:
                        print(f"{BLUE}║{RESET}", end="")
                    else:
                        print(f"{GRAY}│{RESET}", end="")

            print()

            if row < self.board_size - 1:
                for col in range(self.board_size):
                    has_fence = self.horizontal_fences[row][col] or (row > 0 and self.horizontal_fences[row][col-1])
                    if has_fence:
                        print(f"{BLUE}═══{RESET}", end="")
                    else:
                        print(f"{GRAY}───{RESET}", end="")
                    if col < self.board_size - 1:
                        print(f"{GRAY}╬{RESET}", end="")  # Intersection for a cleaner grid
                print()

        print(f"\nPlayer 0 fences left: {self.fences[0]}")
        print(f"Player 1 fences left: {self.fences[1]}")

    def check_winner(self):
        for player, (x, y) in enumerate(self.pawns):
            if player == 0 and y == self.board_size - 1:
                return 0
            if player == 1 and y == 0:
                return 1
        return None


# Example usage
board = QuoridorGame()
board.place_fence(0, (1, 1), 'V')
board.place_fence(1, (7, 1), 'V')
board.place_fence(0, (1, 1), 'H')
board.print_game_state()
print(board.vertical_fences)
# board.move_pawn(0, (1, 4))
# board.move_pawn(1, (7, 4))

# board.print_game_state()


