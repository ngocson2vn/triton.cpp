#include <cstdio>
#include <memory>
#include <random>
#include <atomic>
#include <thread>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "runtime/fprint_mat.h"
#include "runtime/cubin_launcher.h"

using ABType = __nv_half;
using DType = float;

[[maybe_unused]] static std::atomic<int> g_ok(0);
[[maybe_unused]] static std::atomic<int> g_ng(0);
[[maybe_unused]] static std::atomic<int> num_diffs(0);

template <typename ABType, typename DType, int M_TILE, int N_TILE>
void matmul_cpu_tile(ABType* mat_A, ABType* mat_B, DType* mat_D, std::vector<int> a_stride, std::vector<int> b_stride, std::vector<int> d_stride, int M, int N, int K, int m_start, int n_start) {
  static_assert(std::is_same<DType, float>(), "DType must be float");
  DType mat_A_mk;
  DType mat_B_nk;
  for (int i = 0; i < M_TILE; i++) {
    int m = m_start + i;
    for (int j = 0; j < N_TILE; j++) {
      int n = n_start + j;
      int idx = m * d_stride[0] + n * d_stride[1];
      for (int k = 0; k < K; k++) {
        if constexpr(std::is_same<ABType, __nv_half>()) {
          mat_A_mk = __half2float(mat_A[m * a_stride[0] + k * a_stride[1]]);
          mat_B_nk = __half2float(mat_B[n * b_stride[0] + k * b_stride[1]]);
        } else {
          mat_A_mk = __bfloat162float(mat_A[m * a_stride[0] + k * a_stride[1]]);
          mat_B_nk = __bfloat162float(mat_B[n * b_stride[0] + k * b_stride[1]]);
        }

        mat_D[idx] += mat_A_mk * mat_B_nk;
      }
    }
  }
}

template <typename ABType, typename DType, int M_TILE, int N_TILE, int MAX_THREAD_COUNT=1024>
void matmul_cpu_parallel(ABType* mat_A, ABType* mat_B, DType* mat_D, std::vector<int> a_stride, std::vector<int> b_stride, std::vector<int> d_stride, int M, int N, int K) {
  const int M_TILE_COUNT = M / M_TILE;
  const int N_TILE_COUNT = N / N_TILE;
  const int total_tiles = M_TILE_COUNT * N_TILE_COUNT;
  const int total_batches = total_tiles / MAX_THREAD_COUNT + ((total_tiles % MAX_THREAD_COUNT) > 0 ? 1 : 0);
  printf("Total batches of tiles to be processed: %d\n", total_batches);

  std::vector<std::thread> threads;
  int batch_count = 0;
  for (int m_tile = 0; m_tile < M_TILE_COUNT; m_tile++) {
    for (int n_tile = 0; n_tile < N_TILE_COUNT; n_tile++) {
      threads.emplace_back(
        matmul_cpu_tile<ABType, DType, M_TILE, N_TILE>, mat_A, mat_B, mat_D, a_stride, b_stride, d_stride, M, N, K, m_tile * M_TILE, n_tile * N_TILE
      );

      if (threads.size() == MAX_THREAD_COUNT) {
        for (auto& t : threads) {
          t.join();
        }
        batch_count++;
        printf("Done processing batch %d of %zu tiles\n", batch_count, threads.size());
        threads.clear();
      }
    }
  }

  if (threads.size() > 0) {
    for (auto& t : threads) {
      t.join();
    }
    batch_count++;
    printf("Done processing batch %d of %zu tiles\n", batch_count, threads.size());
  }
}

template <typename T, int MAX_NUM_DIFFS=3>
void verify(T* cpu_data, T* gpu_data, int start_idx, int numElements) {
  int ok = 0;
  int ng = 0;
  constexpr T EPSILON = 1.0e-3;
  for (int i = 0; i < numElements; i++) {
    int idx = start_idx + i;
    T diff = std::abs(cpu_data[idx] - gpu_data[idx]);
    if (diff < EPSILON) {
      ok++;
    } else {
      ng++;
      if (num_diffs < MAX_NUM_DIFFS) {
        if (num_diffs == 0) {
          LOG_INFO("Top %d diffs:\n", MAX_NUM_DIFFS);
        }
        LOG_INFO("[idx = %10d] cpu != gpu: %.5f != %.5f (abs diff = %.5f)\n", idx, cpu_data[idx], gpu_data[idx], diff);
        num_diffs++;
      }
    }
  }

  g_ok.fetch_add(ok);
  if (ng > 0) {
    g_ng.fetch_add(ng);
  }
}

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

  std::unique_ptr<ABType> a_mat_ptr(new ABType[M*K]);
  ABType* a_mat = a_mat_ptr.get();

  std::unique_ptr<ABType> b_mat_ptr(new ABType[N*K]);
  ABType* b_mat = b_mat_ptr.get();

  // Initialize input data
  for (int i = 0; i < M*K; i++) a_mat[i] = ABType(dist(gen));
  for (int i = 0; i < N*K; i++) b_mat[i] = ABType(dist(gen));

  // Output file
  FILE* file_ptr = get_file_ptr("output.txt");

  // Dump A and B
  fprint_mat(file_ptr, "A", a_mat, {'m', 'k'}, {M, K}, {K, 1});
  fprintf(file_ptr, "\n\n");
  fprint_mat(file_ptr, "B", b_mat, {'n', 'k'}, {N, K}, {K, 1});
  fprintf(file_ptr, "\n\n");

  std::vector<void*> inputs;
  inputs.push_back(a_mat);
  inputs.push_back(b_mat);

  std::vector<std::size_t> inputSizes;
  inputSizes.push_back(M*K * sizeof(ABType));
  inputSizes.push_back(N*K * sizeof(ABType));

  // Allocate host output
  std::unique_ptr<DType> d_mat_gpu_ptr(new DType[M*N]);
  DType* d_mat_gpu = d_mat_gpu_ptr.get();

  std::vector<void*> outputs;
  std::vector<std::size_t> outputSizes;
  outputs.push_back(d_mat_gpu);
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
  fprint_mat(file_ptr, "D_gpu", d_mat_gpu, {'m', 'n'}, {M, N}, {N, 1});
  fprintf(file_ptr, "\n\n");

  // CPU
  constexpr int MAX_THREAD_COUNT = 1024;
  std::unique_ptr<DType> d_mat_cpu_ptr(new DType[M*N]);
  DType* d_mat_cpu = d_mat_cpu_ptr.get();
  for (int i = 0; i < M*N; i++) d_mat_cpu[i] = DType(0);

  printf("\nRun matmul_cpu_parallel\n");
  matmul_cpu_parallel<ABType, DType, M_TILE, N_TILE, MAX_THREAD_COUNT>(
    a_mat, b_mat, d_mat_cpu,
    {K, 1}, {K, 1}, {N, 1}, M, N, K
  );

  fprint_mat(file_ptr, "D_cpu", d_mat_cpu, {'m', 'n'}, {M, N}, {N, 1});

  // Verify
  printf("\nVerifying matrix D\n");
  std::vector<std::thread> verifyThreads;
  const int kNumElements = (M * N) / MAX_THREAD_COUNT;
  for (int i = 0; i < MAX_THREAD_COUNT; i++) {
    int idx_start = i * kNumElements;
    verifyThreads.emplace_back(
      verify<DType>, 
      d_mat_cpu, d_mat_gpu, idx_start, kNumElements
    );
  }

  for (auto& t : verifyThreads) {
    t.join();
  }

  printf("Matrix D: ok: %d ng: %d\n\n", g_ok.fetch_add(0), g_ng.fetch_add(0));

  printf("Please check output.txt\n");

  return 0;
}
