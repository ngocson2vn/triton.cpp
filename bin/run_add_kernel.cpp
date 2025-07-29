#include <stdio.h>
#include <memory>

#include "runtime/cubin_launcher.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("Usage: %s /path/to/file.cubin KERNEL_NAME BLOCK_SIZE\n", argv[0]);
    return EXIT_FAILURE;
  }

  // Parse arguments
  std::string cubinFile = argv[1];
  std::string kernelName = argv[2];

  int blockSize = -1;
  try {
    blockSize = std::stoi(argv[3]);
  } catch (std::exception& ex) {
    fprintf(stderr, "Failed to parse BLOCK_SIZE, error: %s\n", ex.what());
    return EXIT_FAILURE;
  }

  //
  // Client code
  //

  // Allocate host input
  int numElements = 1024;

  std::unique_ptr<float> h_input1_ptr(new float[numElements]);
  float* h_input1 = h_input1_ptr.get();

  std::unique_ptr<float> h_input2_ptr(new float[numElements]);
  float* h_input2 = h_input2_ptr.get();

  // Initialize input data
  for (int i = 0; i < numElements; i++) {
    h_input1[i] = static_cast<float>(i);
    h_input2[i] = static_cast<float>(i);
  }

  std::vector<void*> inputs;
  inputs.push_back(h_input1);
  inputs.push_back(h_input2);

  std::vector<std::size_t> inputSizes;
  inputSizes.push_back(numElements * sizeof(float));
  inputSizes.push_back(numElements * sizeof(float));

  // Allocate host output
  std::unique_ptr<float> h_output_ptr(new float[numElements]);
  float* h_output = h_output_ptr.get();

  std::vector<void*> outputs;
  std::vector<std::size_t> outputSizes;
  outputs.push_back(h_output);
  outputSizes.push_back(numElements * sizeof(float));

  // An uniform runtime API
  cubin::CubinLauncher launcher(cubinFile);
  bool ok = launcher.launchKernel(kernelName, inputs, inputSizes, outputs, outputSizes, numElements, blockSize);
  if (!ok) {
    return EXIT_FAILURE;
  }

  // Print output
  printf("=====================================\n");
  for (int i = 0; i < numElements; i++) {
    printf("Output[%d] = %f\n", i, h_output[i]);
  }
  printf("=====================================\n");

  return 0;
}
