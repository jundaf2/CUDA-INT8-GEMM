#pragma once
#include <cuda.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <mma.h>
#include <limits>   



constexpr bool GEMM_OP_T = true;
constexpr bool GEMM_OP_N = false;

using namespace nvcuda;
namespace cg = cooperative_groups;

namespace kernel{

  
template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_SIZE_M, int WARP_SIZE_N, int STAGE, bool NoTransA, bool NoTransB, bool RowMajorC>
__global__ void GEMMI8TCU(const int8_t* A, const int8_t* B, int* C, int M, int N, int K)
{
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

  int warp_id = tile32.meta_group_rank();
  int lane_id = tile32.thread_rank();

  constexpr int WARP_SIZE = 32;
  constexpr int TC_SIZE = 16;
  constexpr int WAPR_NUM_N = BLOCK_SIZE_N / WARP_SIZE_N;
  constexpr int WAPR_NUM_M = BLOCK_SIZE_M / WARP_SIZE_M;
  constexpr int WAPR_NUM     = WAPR_NUM_M * WAPR_NUM_N;

  static_assert(NoTransA == GEMM_OP_T, "NoTransA == GEMM_OP_T");
  static_assert(NoTransB == GEMM_OP_N, "NoTransB == GEMM_OP_N");
  static_assert(RowMajorC == GEMM_OP_T, "RowMajorC == GEMM_OP_T");

  __shared__ int8_t SLB[STAGE * (BLOCK_SIZE_K*BLOCK_SIZE_M + BLOCK_SIZE_K*BLOCK_SIZE_N)];

  int8_t* smem_a[2];
  int8_t* smem_b[2];

  smem_a[0] = SLB;
  smem_a[1] = SLB + BLOCK_SIZE_K*BLOCK_SIZE_M;
  smem_b[0] = SLB + STAGE*BLOCK_SIZE_K*BLOCK_SIZE_M;
  smem_b[1] = SLB + STAGE*BLOCK_SIZE_K*BLOCK_SIZE_M + BLOCK_SIZE_K*BLOCK_SIZE_N;

  const int BCM = BLOCK_SIZE_M * blockIdx.y;
  const int BCN = BLOCK_SIZE_N * blockIdx.x;

  const int LDA = NoTransA ? K : M;
  const int LDB = NoTransB ? N : K;
  const int LDC = RowMajorC ? N : M;

  const int WCM = warp_id / WAPR_NUM_N;
  const int WCN = warp_id % WAPR_NUM_N;

  const int BLOCK_K_LOOP = K / BLOCK_SIZE_K;

  const int8_t* BA = A + BCM * LDA;
  const int8_t* BB = B + BCN * LDB;
  int* BC = C + BCM * LDC + BCN;
  int* BWC = BC + WCM * WARP_SIZE_M * LDC + WCN * WARP_SIZE_N;

  constexpr int WARP_M_LOOP = WARP_SIZE_M / TC_SIZE;
  constexpr int WARP_N_LOOP = WARP_SIZE_N / TC_SIZE;
  constexpr int WARP_K_LOOP = BLOCK_SIZE_K / TC_SIZE;

  wmma::fragment<wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::row_major> frag_a[WARP_M_LOOP][WARP_K_LOOP];
  wmma::fragment<wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::col_major> frag_b[WARP_K_LOOP][WARP_N_LOOP];
  wmma::fragment<wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, int> frag_c[WARP_M_LOOP][WARP_N_LOOP];

  #pragma unroll
  for (int i = 0; i < WARP_M_LOOP; i++) {
      #pragma unroll
      for (int j = 0; j < WARP_N_LOOP; j++) {
          wmma::fill_fragment(frag_c[i][j], 0);
      }
  }  

  constexpr int WARP_SIZE_X = 2;
  int lane_id_x = lane_id % (WARP_SIZE_X); // [0,2]
  int lane_id_y = lane_id / (WARP_SIZE_X); // [0,16]

  for(int k=0; k<BLOCK_K_LOOP; k++){
    const auto* load_gmem_addr_a = BA + (warp_id*TC_SIZE + lane_id_y) * LDA + k*BLOCK_SIZE_K + lane_id_x*16;
    const auto* load_gmem_addr_b = BB + (warp_id*TC_SIZE + lane_id_y) * LDB + k*BLOCK_SIZE_K + lane_id_x*16;

    int store_smem_addr_a = __cvta_generic_to_shared(smem_a[k%2] + (warp_id*TC_SIZE + lane_id_y)*BLOCK_SIZE_K + lane_id_x*16);
    int store_smem_addr_b = __cvta_generic_to_shared(smem_b[k%2] + (warp_id*TC_SIZE + lane_id_y)*BLOCK_SIZE_K + lane_id_x*16);
    
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_a), "l"(load_gmem_addr_a), "n"(16));
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(16));
    
    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    
    for(int ki=0; ki<WARP_K_LOOP; ki++)
      for(int yi=0; yi<WARP_M_LOOP; yi++){
        wmma::load_matrix_sync(frag_a[yi][ki], &smem_a[k%2][(WCM*WARP_SIZE_M+yi*TC_SIZE)*BLOCK_SIZE_K+ki*TC_SIZE], BLOCK_SIZE_K);
        for(int xi=0; xi<WARP_N_LOOP; xi++){
          wmma::load_matrix_sync(frag_b[ki][xi], &smem_b[k%2][(WCN*WARP_SIZE_N+xi*TC_SIZE)*BLOCK_SIZE_K+ki*TC_SIZE], BLOCK_SIZE_K);
          wmma::mma_sync(frag_c[yi][xi], frag_a[yi][ki], frag_b[ki][xi], frag_c[yi][xi]);
        }
      }
  }


  #pragma unroll
  for(int yi=0; yi<WARP_M_LOOP; yi++){
    #pragma unroll
    for(int xi=0; xi<WARP_N_LOOP; xi++){
        wmma::store_matrix_sync(BWC + (yi*TC_SIZE)*LDC + xi*TC_SIZE, frag_c[yi][xi], LDC, wmma::mem_row_major);
    }
  }
  
}            

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_SIZE_M, int WARP_SIZE_N, int STAGE, bool NoTransA, bool NoTransB, bool RowMajorC>
__global__ void GEMMI8TCU(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K)
{
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

  int warp_id = tile32.meta_group_rank();
  int lane_id = tile32.thread_rank();

  constexpr int WARP_SIZE = 32;
  constexpr int TC_SIZE = 16;
  constexpr int WAPR_NUM_N = BLOCK_SIZE_N / WARP_SIZE_N;
  constexpr int WAPR_NUM_M = BLOCK_SIZE_M / WARP_SIZE_M;
  constexpr int WAPR_NUM     = WAPR_NUM_M * WAPR_NUM_N;

  static_assert(NoTransA == GEMM_OP_T, "NoTransA == GEMM_OP_T");
  static_assert(NoTransB == GEMM_OP_N, "NoTransB == GEMM_OP_N");
  static_assert(RowMajorC == GEMM_OP_T, "RowMajorC == GEMM_OP_T");

  __shared__ int8_t SLB[STAGE * (BLOCK_SIZE_K*BLOCK_SIZE_M + BLOCK_SIZE_K*BLOCK_SIZE_N)];

  int8_t* smem_a[2];
  int8_t* smem_b[2];

  smem_a[0] = SLB;
  smem_a[1] = SLB + BLOCK_SIZE_K*BLOCK_SIZE_M;
  smem_b[0] = SLB + STAGE*BLOCK_SIZE_K*BLOCK_SIZE_M;
  smem_b[1] = SLB + STAGE*BLOCK_SIZE_K*BLOCK_SIZE_M + BLOCK_SIZE_K*BLOCK_SIZE_N;

  const int BCM = BLOCK_SIZE_M * blockIdx.y;
  const int BCN = BLOCK_SIZE_N * blockIdx.x;

  const int LDA = NoTransA ? K : M;
  const int LDB = NoTransB ? N : K;
  const int LDC = RowMajorC ? N : M;

  const int WCM = warp_id / WAPR_NUM_N;
  const int WCN = warp_id % WAPR_NUM_N;

  const int BLOCK_K_LOOP = K / BLOCK_SIZE_K;

  const int8_t* BA = A + BCM * LDA;
  const int8_t* BB = B + BCN * LDB;
  int8_t* BC = C + BCM * LDC + BCN;
  int8_t* BWC = BC + WCM * WARP_SIZE_M * LDC + WCN * WARP_SIZE_N;

  constexpr int WARP_M_LOOP = WARP_SIZE_M / TC_SIZE;
  constexpr int WARP_N_LOOP = WARP_SIZE_N / TC_SIZE;
  constexpr int WARP_K_LOOP = BLOCK_SIZE_K / TC_SIZE;

  wmma::fragment<wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::row_major> frag_a[WARP_M_LOOP][WARP_K_LOOP];
  wmma::fragment<wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::col_major> frag_b[WARP_K_LOOP][WARP_N_LOOP];
  wmma::fragment<wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, int> frag_c[WARP_M_LOOP][WARP_N_LOOP];

  #pragma unroll
  for (int i = 0; i < WARP_M_LOOP; i++) {
      #pragma unroll
      for (int j = 0; j < WARP_N_LOOP; j++) {
          wmma::fill_fragment(frag_c[i][j], 0);
      }
  }  

  constexpr int WARP_SIZE_X = 2;
  int lane_id_x = lane_id % (WARP_SIZE_X); // [0,2]
  int lane_id_y = lane_id / (WARP_SIZE_X); // [0,16]

  for(int k=0; k<BLOCK_K_LOOP; k++){
    const auto* load_gmem_addr_a = BA + (warp_id*TC_SIZE + lane_id_y) * LDA + k*BLOCK_SIZE_K + lane_id_x*16;
    const auto* load_gmem_addr_b = BB + (warp_id*TC_SIZE + lane_id_y) * LDB + k*BLOCK_SIZE_K + lane_id_x*16;

    int store_smem_addr_a = __cvta_generic_to_shared(smem_a[k%2] + (warp_id*TC_SIZE + lane_id_y)*BLOCK_SIZE_K + lane_id_x*16);
    int store_smem_addr_b = __cvta_generic_to_shared(smem_b[k%2] + (warp_id*TC_SIZE + lane_id_y)*BLOCK_SIZE_K + lane_id_x*16);
    
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_a), "l"(load_gmem_addr_a), "n"(16));
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(16));
    
    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    
    for(int ki=0; ki<WARP_K_LOOP; ki++)
      for(int yi=0; yi<WARP_M_LOOP; yi++){
        wmma::load_matrix_sync(frag_a[yi][ki], &smem_a[k%2][(WCM*WARP_SIZE_M+yi*TC_SIZE)*BLOCK_SIZE_K+ki*TC_SIZE], BLOCK_SIZE_K);
        for(int xi=0; xi<WARP_N_LOOP; xi++){
          wmma::load_matrix_sync(frag_b[ki][xi], &smem_b[k%2][(WCN*WARP_SIZE_N+xi*TC_SIZE)*BLOCK_SIZE_K+ki*TC_SIZE], BLOCK_SIZE_K);
          wmma::mma_sync(frag_c[yi][xi], frag_a[yi][ki], frag_b[ki][xi], frag_c[yi][xi]);
        }
      }
  }

  int gmem_lane_id_x = lane_id % 4; // [0,4]
  int gmem_lane_id_y = lane_id / 4; // [0 8]
  #pragma unroll
  for(int yi=0; yi<WARP_M_LOOP; yi++)
    #pragma unroll
    for(int xi=0; xi<WARP_N_LOOP; xi++)
    {
      int8_t tmp_char[8];
      #pragma unroll 
      for(int i=0;i<8;i++) {
        tmp_char[i] = static_cast<int8_t>(frag_c[yi][xi].x[i]);
      }

      for(int tc_yi=0; tc_yi<2; tc_yi++){
        for(int tc_xi=0; tc_xi<2; tc_xi++){
          auto* store_gmem_addr = reinterpret_cast<char2*>(BWC + (yi*TC_SIZE + tc_yi*TC_SIZE/2 + gmem_lane_id_y) * LDC + xi*TC_SIZE + tc_xi*TC_SIZE/2 + gmem_lane_id_x*2);
          char2 tmp_char2;
          tmp_char2.x = tmp_char[tc_xi*4+tc_yi*2+0];
          tmp_char2.y = tmp_char[tc_xi*4+tc_yi*2+1];
          *store_gmem_addr = tmp_char2; 
        }
      }
    }
}                                


}
