#pragma once
#include <cmath>
#include <iostream>
#include <limits>
#include <assert.h>
#include <cuda.h>

#define ASSERT_CUDA(ret) assert(cudaSuccess==ret)

void print_vec(const int8_t *outv, std::string outn, int start, int end, int row_size) {
  std::cout << outn << ": ";
  for(int i=start; i<end; i++) {
    std::cout << static_cast<float>(outv[i]) << " ";
    if((i-start+1)%row_size==0) std::cout << std::endl;  
  }
  std::cout << std::endl;
}

