# CUDA-INT8-GEMM
The 8-bit GEMM takes two 8-bit input matrices and produces an output matrix which is also of 8-bit.  

C = A*B^T

We adopt the same convention as the cuBLAS library, where the matrices are stored in column-major order. `GEMM_OP_T` means the matrix is transposed in column-major representation, which is equivalent to the non-transposed matrix in row-major representation. `GEMM_OP_N` means the matrix is not transposed in column-major representation, which is equivalent to the transposed matrix in row-major representation. The same convention applies to matrix C.

You may undersand the `T` and `N` in these flags as either `transpose` / `non-transpose` operation for col-major BLAS (Fortran) matrices or  `true` / `not true` for row-major C/C++ matrices.
