#include <stdio.h>

#define N  1023

__global__ void matrixMulGPU( int * a, int * b, int * c, int n)
{
  /*
   * Build out this kernel.
   */
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    int stride_row = blockDim.x * gridDim.x;
    int stride_col = blockDim.y * gridDim.y;

    for (int i = row; i < n; i+=stride_row ){
        for (int j = col; j < n; j+=stride_col){
            int val = 0;
            for (int k = 0; k < n; k +=1 ){
                val += a[row * n + k] * b[k * n + col];
            }
        c[row * n + col] = val;
        }
    }

}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */

void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

int main()
{
  int *a, *b, *c_cpu, *c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  // Initialize memory; create 2D matrices
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  /*
   * Assign `threads_per_block` and `number_of_blocks` 2D values
   * that can be used in matrixMulGPU above.
   */

  dim3 threads_per_block(32, 32, 1);
  dim3 number_of_blocks(((N+threads_per_block.x-1)/threads_per_block.x), (N+threads_per_block.y-1)/threads_per_block.y);  

  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu, N );

  cudaDeviceSynchronize();

  // Call the CPU version to check our work
  matrixMulCPU( a, b, c_cpu );

  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("True answer: %d, Wrong answer %d", c_cpu[row * N + col], c_gpu[row * N + col]);
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  // Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}