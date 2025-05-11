#include <torch/script.h>

#include <iostream>
#include <memory>
#include <vector>
#include <tuple>

// Function to convert a game state to tensors for model input
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> game_state_to_tensors(
    const std::tuple<
        std::pair<std::pair<int, int>, std::pair<int, int>>,  // player positions
        std::pair<int, int>,                                  // fence counts
        std::vector<std::vector<bool>>,                       // horizontal fences
        std::vector<std::vector<bool>>,                       // vertical fences
        int,                                                  // current player
        int                                                   // move count
    >& state) {
    
    const auto& positions = std::get<0>(state);
    const auto& fence_remaining = std::get<1>(state);
    const auto& h_fences = std::get<2>(state);
    const auto& v_fences = std::get<3>(state);
    const int current_player = std::get<4>(state);
    const int moves = std::get<5>(state);
    
    // Create a 4x17x17 tensor to represent the board state
    // Channels: 0=player0, 1=player1, 2=horizontal fences, 3=vertical fences
    torch::Tensor board_tensor = torch::zeros({1, 4, 17, 17});
    
    // Set player positions
    // Player 0
    int player0_row = positions.first.first;
    int player0_col = positions.first.second;
    board_tensor[0][0][player0_row][player0_col] = 1.0;
    
    // Player 1
    int player1_row = positions.second.first;
    int player1_col = positions.second.second;
    board_tensor[0][1][player1_row][player1_col] = 1.0;
    
    // Set horizontal fences
    const int fence_size = h_fences.size();
    for (int row = 0; row < fence_size; row++) {
        for (int col = 0; col < fence_size; col++) {
            if (h_fences[row][col]) {
                // Set fence in position and extend it horizontally
                board_tensor[0][2][row][col] = 1.0;
                if (col + 1 < 17) { 
                    board_tensor[0][2][row][col + 1] = 1.0;
                }
            }
        }
    }
    
    // Set vertical fences
    for (int row = 0; row < fence_size; row++) {
        for (int col = 0; col < fence_size; col++) {
            if (v_fences[row][col]) {
                // Set fence in position and extend it vertically
                board_tensor[0][3][row][col] = 1.0;
                if (row + 1 < 17) {
                    board_tensor[0][3][row + 1][col] = 1.0;
                }
            }
        }
    }
    
    // Create fence counts tensor
    torch::Tensor fence_counts = torch::zeros({1, 2});
    fence_counts[0][0] = fence_remaining.first;   // Player 0 fences
    fence_counts[0][1] = fence_remaining.second;  // Player 1 fences
    
    // Create move count tensor
    torch::Tensor move_count = torch::zeros({1, 1});
    move_count[0][0] = moves;
    
    return {board_tensor, fence_counts, move_count};
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

  // Example game state (positions, fence counts, h_fences, v_fences, current player, move count)
  // Initial state with players at their starting positions
  auto game_state = std::make_tuple(
      std::make_pair(std::make_pair(0, 4), std::make_pair(8, 4)),  // player positions
      std::make_pair(10, 10),                                      // fence counts
      std::vector<std::vector<bool>>(8, std::vector<bool>(8, false)), // h_fences 8x8
      std::vector<std::vector<bool>>(8, std::vector<bool>(8, false)), // v_fences 8x8
      0,                                                           // current player (0)
      0                                                            // move count
  );
  
  // Convert game state to tensor inputs
  auto [board_tensor, fence_counts, move_count] = game_state_to_tensors(game_state);

  // Pack them into an inner tuple
  auto inner_tuple =
      torch::ivalue::Tuple::create({board_tensor, fence_counts, move_count});

  // Then wrap that in an outer tuple to match the Python ( (a, b, c), )
  // structure
  std::vector<torch::IValue> inputs;
  inputs.push_back(inner_tuple);

  // Run inference
  torch::Tensor output = module.forward(inputs).toTensor();

  // Optional: print or inspect output
  std::cout << output << std::endl;

  std::cout << "ok\n";
}
