#include <torch/script.h>

#include <iostream>
#include <memory>

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

  // Create your input tensors
  torch::Tensor board_tensor = torch::zeros({1, 4, 17, 17});
  torch::Tensor fence_counts = torch::zeros({1, 2});
  torch::Tensor move_count = torch::zeros({1, 1});

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
