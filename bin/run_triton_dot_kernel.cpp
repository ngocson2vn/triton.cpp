#include <cstdio>
#include <memory>
#include <random>

#include <cuda_fp16.h>
#include "runtime/fprint_mat.h"
#include "runtime/cubin_launcher.h"

using ABType = __nv_half;
using DType = float;

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Usage: %s /path/to/file.cubin\n", argv[0]);
    return EXIT_FAILURE;
  }

  // Parse arguments
  const std::string cubinFile = argv[1];

  // Constants
  constexpr int M_TILE = 64;
  constexpr int N_TILE = 32;
  constexpr int K_TILE = 64;
  constexpr int M = M_TILE;
  constexpr int N = N_TILE;
  constexpr int K = K_TILE;
  constexpr int blockSize = 128;
  const std::string kernelName = "triton_dot";

  //
  // Client code
  //

  // Allocate host input
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  
  // For debugging:
  // std::uniform_int_distribution<int> dist(0, 3);

  std::uniform_real_distribution<float> dist(0.0, 0.1);

  std::unique_ptr<ABType> h_mat1_ptr(new ABType[M*K]);
  ABType* h_mat1 = h_mat1_ptr.get();

  std::unique_ptr<ABType> h_mat2_ptr(new ABType[N*K]);
  ABType* h_mat2 = h_mat2_ptr.get();

  // Initialize input data
  for (int i = 0; i < M*K; i++) h_mat1[i] = ABType(dist(gen));
  for (int i = 0; i < N*K; i++) h_mat2[i] = ABType(dist(gen));

  // Output file
  FILE* file_ptr = get_file_ptr("output.txt");

  // Dump mat1 and mat2
  fprint_mat(file_ptr, "A", h_mat1, {'m', 'k'}, {M, K}, {K, 1});
  fprintf(file_ptr, "\n\n");
  fprint_mat(file_ptr, "B", h_mat2, {'n', 'k'}, {N, K}, {K, 1});
  fprintf(file_ptr, "\n\n");

  std::vector<void*> inputs;
  inputs.push_back(h_mat1);
  inputs.push_back(h_mat2);

  std::vector<std::size_t> inputSizes;
  inputSizes.push_back(M*K * sizeof(ABType));
  inputSizes.push_back(N*K * sizeof(ABType));

  // Allocate host output
  std::unique_ptr<DType> h_output_ptr(new DType[M*N]);
  DType* h_output = h_output_ptr.get();

  std::vector<void*> outputs;
  std::vector<std::size_t> outputSizes;
  outputs.push_back(h_output);
  outputSizes.push_back(M*N * sizeof(DType));

  // An uniform runtime API
  rt::CubinLauncher launcher(cubinFile);
  bool ok = launcher.launchMMAKernel<ABType, DType, M_TILE, N_TILE, K_TILE>(
    kernelName, 
    inputs, inputSizes, 
    outputs, outputSizes, 
    M, N, K, blockSize
  );

  if (!ok) {
    return EXIT_FAILURE;
  }

  // Print output
  printf("Dumping D matrix\n");
  fprint_mat(file_ptr, "D", h_output, {'m', 'n'}, {M, N}, {N, 1});
  printf("Please check output.txt\n");

  return 0;
}
