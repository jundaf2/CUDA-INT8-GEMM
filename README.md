# CUDA-INT8-GEMM
## Introduction
The 8-bit GEMM takes two 8-bit input matrices and produces an output matrix which is also of 8-bit.  

C = A*B^T

We adopt the same convention as the cuBLAS library, where the matrices are stored in column-major order. `GEMM_OP_T` means the matrix is transposed in column-major representation, which is equivalent to the non-transposed matrix in row-major representation. `GEMM_OP_N` means the matrix is not transposed in column-major representation, which is equivalent to the transposed matrix in row-major representation. The same convention applies to matrix C.

You may undersand the `T` and `N` in these flags as either `transpose` / `non-transpose` operation for col-major BLAS (Fortran) matrices or  `true` / `not true` for row-major C/C++ matrices.


## Current support

The output type can be either `int8` or `int32` depending on your usage. For example, when you use GEMM in a 8-bit framework, you may want to use `int8` output as the input of next layer's operation. The tensor core itself uses `int32` as accumalator, which is the same as the cuBLAS library.

Current
``` example cmd for int8 output:
    ./test_gemm_i8 1024 1024 32 1 0 1 1 1
```

``` example cmd for int32 output:
    ./test_gemm_i8 1024 1024 32 1 0 1 1 0
```