// System includes
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "cpu_gemm.h"
#include "util.h"
#include "gemm_i8.cuh"

using namespace std;


template <bool use_tcu>
void GEMMI8(cudaStream_t stream, 
            const int8_t *A, const int8_t *B, int8_t *C,
            int M, int N, int K,
            bool transA, bool transB, bool transC) 
{
  constexpr int BLOCK_M = 256;
  constexpr int BLOCK_N = 256;
  constexpr int BLOCK_K = 32;
  constexpr int WARP_M = 64;
  constexpr int WARP_N = 64;
  constexpr int WARP_SIZE = 32;

  dim3 block((BLOCK_M / WARP_M) * (BLOCK_N / WARP_N) * WARP_SIZE, 1, 1);   
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N / 1, (M + BLOCK_M - 1) / BLOCK_M);  
  
  if(transA==GEMM_OP_T && transB==GEMM_OP_N && transC==GEMM_OP_T)
    kernel::GEMMI8TCU<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, 2, GEMM_OP_T, GEMM_OP_N, GEMM_OP_T><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  
}


class GEMM {
public:
  GEMM(bool use_tcu, int m, int n, int k, bool transa, bool transb, bool transc) {
    this->use_tcu = use_tcu;
    this->M = m;
    this->N = n;
    this->K = k;

    this->len_a = M*K;
    this->len_b = N*K;
    this->len_c = M*N;

    this->trans_a = transa;
    this->trans_b = transb;
    this->trans_c = transc;

    cout << "compute type=int32" << ", "
          << "data type=int8" << ", "
          << "use_tcu=" << use_tcu << ", "
          << "M=" << m << ", "
          << "N=" << n << ", "
          << "K=" << k
          << endl;

    generateTestData();
  }

  ~GEMM()  = default;

  void generateTestData() {
    
    const auto random_seed = 2023;
    std::mt19937 generator(static_cast<unsigned int>(random_seed));

    h_mat_A = vector<int8_t>(len_a, 0);
    h_mat_B = vector<int8_t>(len_b, 0);
    h_mat_C = vector<int8_t>(len_c, 0);
    h_mat_C_ref = vector<int8_t>(len_c, 0);

    std::uniform_int_distribution<> uniform_char_distribution(CHAR_MIN, CHAR_MAX);

    auto rand_gen = std::bind(uniform_char_distribution, generator);
    auto const_gen = []() { return 1; };

    generate_n(h_mat_A.begin(), len_a, const_gen);
    generate_n(h_mat_B.begin(), len_b, const_gen);

  }

public:
  void testGEMM() {
    cudaStream_t stream;
    ASSERT_CUDA(cudaStreamCreate(&stream));

    // CPU reference
    {
      cpuGEMM<float, float, int8_t, int8_t>(
          h_mat_A.data(), h_mat_B.data(), h_mat_C_ref.data(), M, N, K,
          len_a, len_b, len_c, 1, static_cast<float>(1), static_cast<float>(0), 
          GEMM_OP_T, GEMM_OP_N, GEMM_OP_T);
    }

    ASSERT_CUDA(cudaMalloc(&d_mat_A, len_a * sizeof(int8_t))); 
    ASSERT_CUDA(cudaMalloc(&d_mat_B, len_b * sizeof(int8_t)));
    ASSERT_CUDA(cudaMalloc(&d_mat_C, len_c * sizeof(int8_t)));

    ASSERT_CUDA(cudaMemcpy(d_mat_A, h_mat_A.data(), len_a * sizeof(int8_t), cudaMemcpyHostToDevice)); 
    ASSERT_CUDA(cudaMemcpy(d_mat_B, h_mat_B.data(), len_b * sizeof(int8_t), cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemset(d_mat_C, 0, len_c * sizeof(int8_t)));

    // warp up the device

    // time it
    float milliseconds = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    {  
      if(use_tcu) GEMMI8<true>(stream, d_mat_A, d_mat_B, d_mat_C, M, N, K, trans_a, trans_b, trans_c);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds , start, stop);
    
    double   flops = (double)(M*N*K*2)*1.0;
    double   gigaFlops = (flops * 1.0e-9f) / (milliseconds  / 1000.0f);
    double   bandWidth = (double)(len_a+len_b+len_c)*sizeof(int8_t) / (milliseconds  * 1000 * 1000);
    printf("\033[31;47m INT8 GEMM took %.3f ms, %.2f GFlop/s, %.2f GB/s \033[0m\n", milliseconds , gigaFlops, bandWidth);
    ASSERT_CUDA(cudaDeviceSynchronize());
    ASSERT_CUDA(cudaEventDestroy(start));
    ASSERT_CUDA(cudaEventDestroy(stop));
    ASSERT_CUDA(cudaMemcpy(h_mat_C.data(), d_mat_C, len_c * sizeof(int8_t), cudaMemcpyDeviceToHost));
    
    ASSERT_CUDA(cudaFree(d_mat_A));
    ASSERT_CUDA(cudaFree(d_mat_B));
    ASSERT_CUDA(cudaFree(d_mat_C));
    ASSERT_CUDA(cudaStreamDestroy(stream));

    print_vec(h_mat_C.data(), "h_mat_C: ", 0, N, N);
    print_vec(h_mat_C_ref.data(), "h_mat_C_ref: ", 0, N, N);

    if(h_mat_C == h_mat_C_ref) {
      cout << "test passed !" << endl;
    } else {
      cout << "test failed !" << endl;
    }
  }

protected:

  bool use_tcu;
  int M, N, K;
  int len_a, len_b, len_c;
  bool trans_a, trans_b, trans_c;

  vector<int8_t> h_mat_A;
  vector<int8_t> h_mat_B;
  vector<int8_t> h_mat_C;
  vector<int8_t> h_mat_C_ref;

  int8_t *d_mat_A;
  int8_t *d_mat_B;
  int8_t *d_mat_C;
};


int main(int argc, char **argv) {
  int M = 256;
  int N = 256;
  int K = 32;

  bool trans_a = GEMM_OP_T;
  bool trans_b = GEMM_OP_N;
  bool trans_c = GEMM_OP_T;

  bool use_tcu = true;

  if(argc > 1) {
    M = atoi(argv[1]);
  }
  if(argc > 2) {
    N = atoi(argv[2]);
  }
  if(argc > 3) {
    K = atoi(argv[3]);
  }
  if(argc > 4) {
    trans_a = atoi(argv[4]);
  }
  if(argc > 5) {
    trans_b = atoi(argv[5]);
  }
  if(argc > 6) {
    trans_c = atoi(argv[6]);
  }



  GEMM gemm(use_tcu, M, N, K, trans_a, trans_b, trans_c);
  gemm.testGEMM();

  return 0;
}