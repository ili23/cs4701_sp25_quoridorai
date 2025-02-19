import unittest
from game_engine.game_state import QuoridorGame

class TestQuoridorGame(unittest.TestCase):
    def setUp(self):
        """Set up a new game instance before each test"""
        self.game = QuoridorGame()

    def test_initial_state(self):
        """Test the initial game state"""
        self.assertEqual(self.game.board_size, 9)
        self.assertEqual(self.game.pawns, [(4, 0), (4, 8)])
        self.assertEqual(self.game.fences, [10, 10])
        self.assertEqual(self.game.current_player, 0)

    def test_valid_pawn_moves(self):
        """Test valid pawn movements"""
        # Test basic moves in all directions for player 1
        valid_moves = [(4, 1), (3, 0), (5, 0)]
        for move in valid_moves:
            with self.subTest(move=move):
                game = QuoridorGame()  # Fresh game for each move
                game.move_pawn(0, move)
                self.assertEqual(game.pawns[0], move)

    def test_invalid_pawn_moves(self):
        """Test invalid pawn movements"""
        invalid_moves = [
            (4, 2),  # Too far
            (-1, 0),  # Off board
            (9, 0),  # Off board
            (4, 8),  # Occupied by player 2
        ]
        for move in invalid_moves:
            with self.subTest(move=move):
                with self.assertRaises(ValueError):
                    self.game.move_pawn(0, move)

    def test_fence_placement(self):
        """Test valid and invalid fence placements"""
        # Valid horizontal fence
        self.game.place_fence(0, (0, 0), 'h')
        self.assertTrue(self.game.horizontal_fences[0][0])
        self.assertTrue(self.game.horizontal_fences[0][1])
        self.assertEqual(self.game.fences[0], 9)

        # Valid vertical fence
        self.game.place_fence(0, (0, 0), 'v')
        self.assertTrue(self.game.vertical_fences[0][0])
        self.assertTrue(self.game.vertical_fences[1][0])
        self.assertEqual(self.game.fences[0], 8)

        # Invalid fence placement (overlapping)
        with self.assertRaises(ValueError):
            self.game.place_fence(0, (0, 0), 'h')

    def test_jumping_mechanics(self):
        """Test pawn jumping mechanics"""
        # Set up a situation where jumping is possible
        self.game.pawns = [(4, 3), (4, 4)]  # Players adjacent to each other
        
        # Test valid forward jump
        self.game.move_pawn(0, (4, 5))
        self.assertEqual(self.game.pawns[0], (4, 5))

        # Reset positions
        self.game.pawns = [(4, 4), (4, 3)]
        
        # Test diagonal jump when blocked
        self.game.horizontal_fences[4][4] = True  # Place blocking fence
        self.game.move_pawn(0, (5, 4))  # Should allow diagonal jump
        self.assertEqual(self.game.pawns[0], (5, 4))

    def test_win_condition(self):
        """Test win condition detection"""
        # No winner in initial state
        self.assertIsNone(self.game.check_win_condition())

        # Player 1 wins
        self.game.pawns = [(4, 8), (4, 7)]
        self.assertEqual(self.game.check_win_condition(), 0)

        # Player 2 wins
        self.game.pawns = [(4, 1), (4, 0)]
        self.assertEqual(self.game.check_win_condition(), 1)

    def test_path_availability(self):
        """Test path checking functionality"""
        # Initial state should have paths available
        self.assertTrue(self.game.is_path_available())

        # Create a blocking scenario
        game = QuoridorGame()
        # Place fences to block player 1
        game.place_fence(0, (3, 0), 'h')
        game.place_fence(0, (5, 0), 'h')
        game.place_fence(0, (4, 1), 'v')
        game.place_fence(0, (4, -1), 'v')
        
        # Should detect no available path
        self.assertFalse(game.is_path_available())

    def test_board_visualization(self):
        """Test the board visualization with some moves and fences"""
        game = QuoridorGame()
        # Place some fences
        game.place_fence(0, (2, 2), 'h')
        game.place_fence(1, (3, 3), 'v')
        # Move a pawn
        game.move_pawn(0, (4, 1))
        # Print the board
        game.print_game_state()

    def test_invalid_fence_positions(self):
        """Test fence placement with invalid positions"""
        invalid_fence_positions = [
            ((0, 0), (2, 0)),  # Non-adjacent horizontal
            ((0, 0), (0, 2)),  # Non-adjacent vertical
            ((0, 0), (1, 1)),  # Diagonal
            ((8, 8), (8, 9)),  # Out of bounds
            ((0, 0), (0, 0)),  # Same position
        ]
        
        for pos1, pos2 in invalid_fence_positions:
            with self.subTest(pos1=pos1, pos2=pos2):
                with self.assertRaises(ValueError):
                    self.game.place_fence(0, pos1, pos2)

    def test_is_fence_between(self):
        # Place a horizontal fence
        self.game.horizontal_fences[2][4] = True
        self.assertTrue(self.game.is_fence_between((4, 2), (4, 3)))
        self.assertFalse(self.game.is_fence_between((4, 3), (4, 4)))

        # Place a vertical fence
        self.game.vertical_fences[3][3] = True
        self.assertTrue(self.game.is_fence_between((3, 3), (4, 3)))
        self.assertFalse(self.game.is_fence_between((4, 3), (5, 3)))

if __name__ == '__main__':
    unittest.main() 