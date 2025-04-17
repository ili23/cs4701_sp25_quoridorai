import unittest
import game_engine.game_state_tuple as gs

class TestGameState(unittest.TestCase):
    def setUp(self):
        """Create a fresh initial state for each test"""
        self.initial_state = gs.create_initial_state()
        # Unpack state for easier access in tests
        self.pawns, self.fences, self.h_fences, self.v_fences, self.current_player = self.initial_state

    def test_create_initial_state(self):
        """Test the initial state creation"""
        state = gs.create_initial_state()
        pawns, fences, h_fences, v_fences, current_player = state
        
        # Check initial pawn positions
        self.assertEqual(pawns, ((0, 4), (8, 4)))
        
        # Check initial fence counts
        self.assertEqual(fences, (10, 10))
        
        # Check fence layouts (all False)
        for row in range(gs.BOARD_SIZE - 1):
            for col in range(gs.BOARD_SIZE - 1):
                self.assertFalse(h_fences[row][col])
                self.assertFalse(v_fences[row][col])
        
        # Check initial player
        self.assertEqual(current_player, gs.PLAYER_ONE)

    def test_get_current_player(self):
        """Test getting the current player"""
        # Test with initial state (player 0)
        self.assertEqual(gs.get_current_player(self.initial_state), 0)
        
        # Test with modified state (player 1)
        modified_state = (self.pawns, self.fences, self.h_fences, self.v_fences, 1)
        self.assertEqual(gs.get_current_player(modified_state), 1)

    def test_is_fence_between_no_fence(self):
        """Test is_fence_between when no fence exists"""
        # Adjacent positions without a fence between
        self.assertFalse(gs.is_fence_between(self.initial_state, (4, 4), (4, 5)))
        self.assertFalse(gs.is_fence_between(self.initial_state, (4, 4), (5, 4)))

    def test_is_fence_between_with_fence(self):
        """Test is_fence_between when fence exists"""
        # Create a state with fences for testing
        h_fences_list = [list(row) for row in self.h_fences]
        v_fences_list = [list(row) for row in self.v_fences]
        
        # Place a horizontal fence at (4, 4)
        h_fences_list[4][4] = True
        h_fences = tuple(tuple(row) for row in h_fences_list)
        
        # Place a vertical fence at (2, 3)
        v_fences_list[2][3] = True
        v_fences = tuple(tuple(row) for row in v_fences_list)
        
        fence_state = (self.pawns, self.fences, h_fences, v_fences, self.current_player)
        
        # Test horizontal fence blocking vertical movement
        self.assertTrue(gs.is_fence_between(fence_state, (4, 4), (5, 4)))
        self.assertTrue(gs.is_fence_between(fence_state, (4, 5), (5, 5)))  # Extended fence
        
        # Test vertical fence blocking horizontal movement
        self.assertTrue(gs.is_fence_between(fence_state, (2, 3), (2, 4)))
        self.assertTrue(gs.is_fence_between(fence_state, (3, 3), (3, 4)))  # Extended fence

    def test_is_fence_between_out_of_bounds(self):
        """Test is_fence_between with out of bounds positions"""
        # Out of bounds is treated as blocked
        self.assertTrue(gs.is_fence_between(self.initial_state, (0, 0), (-1, 0)))
        self.assertTrue(gs.is_fence_between(self.initial_state, (8, 8), (8, 9)))

    def test_is_fence_between_non_adjacent(self):
        """Test is_fence_between with non-adjacent positions"""
        # Non-adjacent positions are treated as blocked
        self.assertTrue(gs.is_fence_between(self.initial_state, (0, 0), (2, 2)))
        self.assertTrue(gs.is_fence_between(self.initial_state, (4, 4), (4, 6)))

    def test_get_valid_pawn_moves_initial(self):
        """Test valid pawn moves from initial position"""
        # Player 0 in initial position (0, 4)
        moves = gs.get_valid_pawn_moves(self.initial_state, 0)
        expected_moves = [
            ("move", (0, 5)),  # Right
            ("move", (1, 4)),  # Down
            ("move", (0, 3))   # Left
        ]
        # Same length and contain same moves (order might differ)
        self.assertEqual(len(moves), len(expected_moves))
        for move in expected_moves:
            self.assertIn(move, moves)
        
        # Player 1 in initial position (8, 4)
        moves = gs.get_valid_pawn_moves(self.initial_state, 1)
        expected_moves = [
            ("move", (8, 5)),  # Right
            ("move", (7, 4)),  # Up
            ("move", (8, 3))   # Left
        ]
        self.assertEqual(len(moves), len(expected_moves))
        for move in expected_moves:
            self.assertIn(move, moves)

    def test_get_valid_pawn_moves_with_fence(self):
        """Test valid pawn moves with fences blocking some directions"""
        # Create a state with fences for testing
        h_fences_list = [list(row) for row in self.h_fences]
        v_fences_list = [list(row) for row in self.v_fences]
        
        # Place a horizontal fence below player 0
        h_fences_list[0][4] = True
        h_fences = tuple(tuple(row) for row in h_fences_list)
        
        fence_state = (self.pawns, self.fences, h_fences, self.v_fences, self.current_player)
        
        # Player 0 should not be able to move down
        moves = gs.get_valid_pawn_moves(fence_state, 0)
        expected_moves = [
            ("move", (0, 5)),  # Right
            ("move", (0, 3))   # Left
        ]
        self.assertEqual(len(moves), len(expected_moves))
        for move in expected_moves:
            self.assertIn(move, moves)
        self.assertNotIn(("move", (1, 4)), moves)  # Down should be blocked

    def test_get_valid_pawn_moves_jump(self):
        """Test jumping over opponent"""
        # Create a state where pawns are adjacent
        pawns = ((4, 4), (5, 4))  # Player 0 above player 1
        state = (pawns, self.fences, self.h_fences, self.v_fences, self.current_player)
        
        # Player 0 should be able to jump over player 1
        moves = gs.get_valid_pawn_moves(state, 0)
        self.assertIn(("move", (6, 4)), moves)  # Jump straight over

    def test_get_valid_pawn_moves_jump_diagonal(self):
        """Test diagonal jumping when straight jump is blocked"""
        # Create pawns where they're adjacent
        pawns = ((4, 4), (5, 4))  # Player 0 above player 1
        
        # Create a fence that blocks straight jump
        h_fences_list = [list(row) for row in self.h_fences]
        h_fences_list[5][4] = True  # Fence below player 1
        h_fences = tuple(tuple(row) for row in h_fences_list)
        
        state = (pawns, self.fences, h_fences, self.v_fences, self.current_player)
        
        # Player 0 should be able to jump diagonally
        moves = gs.get_valid_pawn_moves(state, 0)
        self.assertIn(("move", (5, 3)), moves)  # Jump left
        self.assertIn(("move", (5, 5)), moves)  # Jump right
        self.assertNotIn(("move", (6, 4)), moves)  # Straight jump blocked

    def test_path_exists(self):
        """Test path existence to goal row"""
        # Initially, paths should exist for both players
        self.assertTrue(gs.path_exists(self.initial_state, 0, gs.BOARD_SIZE - 1))
        self.assertTrue(gs.path_exists(self.initial_state, 1, 0))
        
        # Create a state with a fence that blocks player 0's path
        h_fences_list = [list(row) for row in self.h_fences]
        
        # Create a horizontal wall across the entire board
        for col in range(gs.BOARD_SIZE - 1):
            h_fences_list[7][col] = True
        h_fences = tuple(tuple(row) for row in h_fences_list)
        
        blocked_state = (self.pawns, self.fences, h_fences, self.v_fences, self.current_player)
        
        # Player 0 should not have a path
        self.assertFalse(gs.path_exists(blocked_state, 0, gs.BOARD_SIZE - 1))
        # Player 1 should still have a path
        self.assertTrue(gs.path_exists(blocked_state, 1, 0))

    def test_path_exists_for_both_players(self):
        """Test that both players have paths to their goals"""
        # Initially, both players should have paths
        self.assertTrue(gs.path_exists_for_both_players(self.initial_state))
        
        # Create a state with a fence that blocks player 0's path
        h_fences_list = [list(row) for row in self.h_fences]
        
        # Create a horizontal wall across the entire board
        for col in range(gs.BOARD_SIZE - 1):
            h_fences_list[7][col] = True
        h_fences = tuple(tuple(row) for row in h_fences_list)
        
        blocked_state = (self.pawns, self.fences, h_fences, self.v_fences, self.current_player)
        
        # With player 0 blocked, both should not have paths
        self.assertFalse(gs.path_exists_for_both_players(blocked_state))

    def test_is_valid_fence_placement(self):
        """Test valid fence placement"""
        # Valid horizontal fence placement
        self.assertTrue(gs.is_valid_fence_placement(self.initial_state, (4, 4), 'H'))
        
        # Valid vertical fence placement
        self.assertTrue(gs.is_valid_fence_placement(self.initial_state, (4, 4), 'V'))
        
        # Invalid - out of bounds
        self.assertFalse(gs.is_valid_fence_placement(self.initial_state, (8, 8), 'H'))
        self.assertFalse(gs.is_valid_fence_placement(self.initial_state, (8, 8), 'V'))
        
        # Create a state with existing fences
        h_fences_list = [list(row) for row in self.h_fences]
        v_fences_list = [list(row) for row in self.v_fences]
        
        h_fences_list[4][4] = True
        v_fences_list[2][3] = True
        
        h_fences = tuple(tuple(row) for row in h_fences_list)
        v_fences = tuple(tuple(row) for row in v_fences_list)
        
        fence_state = (self.pawns, self.fences, h_fences, v_fences, self.current_player)
        
        # Invalid - fence already exists at position
        self.assertFalse(gs.is_valid_fence_placement(fence_state, (4, 4), 'H'))
        self.assertFalse(gs.is_valid_fence_placement(fence_state, (2, 3), 'V'))
        
        # Invalid - crossing fences
        self.assertFalse(gs.is_valid_fence_placement(fence_state, (2, 3), 'H'))
        
        # Test fence that would block all paths
        h_fences_list = [list(row) for row in self.h_fences]
        
        # Create almost a full horizontal wall
        for col in range(gs.BOARD_SIZE - 2):
            h_fences_list[4][col] = True
        h_fences = tuple(tuple(row) for row in h_fences_list)
        
        almost_blocked_state = (self.pawns, self.fences, h_fences, self.v_fences, self.current_player)
        
        # The last fence that would complete the blockade should be invalid
        self.assertFalse(gs.is_valid_fence_placement(almost_blocked_state, (4, gs.BOARD_SIZE - 2), 'H'))

    def test_get_valid_fence_moves(self):
        """Test getting valid fence moves"""
        # Test initial state
        moves = gs.get_valid_fence_moves(self.initial_state, 0)
        # Should be many valid fence moves in initial state
        self.assertGreater(len(moves), 0)
        
        # Test with no fences left
        no_fences_state = (self.pawns, (0, 10), self.h_fences, self.v_fences, self.current_player)
        moves = gs.get_valid_fence_moves(no_fences_state, 0)
        self.assertEqual(len(moves), 0)  # No moves should be available

    def test_get_possible_moves(self):
        """Test getting all possible moves for current player"""
        # Initial state, player 0
        moves = gs.get_possible_moves(self.initial_state)
        
        # Should contain both pawn and fence moves
        pawn_moves = [m for m in moves if m[0] == "move"]
        fence_moves = [m for m in moves if m[0] == "fence"]
        
        self.assertGreater(len(pawn_moves), 0)
        self.assertGreater(len(fence_moves), 0)
        
        # Test with no fences left
        no_fences_state = (self.pawns, (0, 10), self.h_fences, self.v_fences, self.current_player)
        moves = gs.get_possible_moves(no_fences_state)
        
        # Should only contain pawn moves
        self.assertTrue(all(m[0] == "move" for m in moves))

    def test_apply_move_pawn(self):
        """Test applying a pawn move"""
        # Move player 0 down
        move = ("move", (1, 4))
        new_state = gs.apply_move(self.initial_state, move)
        
        # Check pawn position updated
        self.assertEqual(new_state[0][0], (1, 4))
        # Check player switched
        self.assertEqual(new_state[4], 1)

    def test_apply_move_fence(self):
        """Test applying a fence move"""
        # Place horizontal fence
        move = ("fence", (4, 4), "H")
        new_state = gs.apply_move(self.initial_state, move)
        
        # Check fence placed
        self.assertTrue(new_state[2][4][4])
        # Check fence count decreased
        self.assertEqual(new_state[1][0], 9)
        # Check player switched
        self.assertEqual(new_state[4], 1)

    def test_check_winner(self):
        """Test checking for a winner"""
        # No winner in initial state
        self.assertIsNone(gs.check_winner(self.initial_state))
        
        # Player 0 wins by reaching bottom row
        winning_pawns = ((8, 4), (1, 4))
        winning_state = (winning_pawns, self.fences, self.h_fences, self.v_fences, self.current_player)
        self.assertEqual(gs.check_winner(winning_state), 0)
        
        # Player 1 wins by reaching top row
        winning_pawns = ((7, 4), (0, 4))
        winning_state = (winning_pawns, self.fences, self.h_fences, self.v_fences, self.current_player)
        self.assertEqual(gs.check_winner(winning_state), 1)

    def test_is_terminal(self):
        """Test terminal state detection"""
        # Initial state is not terminal
        self.assertFalse(gs.is_terminal(self.initial_state))
        
        # Winning state is terminal
        winning_pawns = ((8, 4), (1, 4))
        winning_state = (winning_pawns, self.fences, self.h_fences, self.v_fences, self.current_player)
        self.assertTrue(gs.is_terminal(winning_state))


if __name__ == '__main__':
    unittest.main()
