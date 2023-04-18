#include <stdio.h>

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */
__global__
void initWith(float num, float *a, int N)
{
  
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < N) { a[index] = num; }

}

/*
 * Device kernel stores into `result` the sum of each
 * same-indexed value of `a` and `b`.
 */

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

/*
 * Host function to confirm values in `vector`. This function
 * assumes all values are the same `target` value.
 */

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  /*
   * nsys should register performance changes when execution configuration
   * is updated.
   */

  threadsPerBlock = 1024;
  numberOfBlocks = 40;

  int deviceId;
  cudaGetDevice(&deviceId);
  
  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);

  initWith<<<numberOfBlocks*N/threadsPerBlock+1, threadsPerBlock>>>(3, a, N);
  initWith<<<numberOfBlocks*N/threadsPerBlock+1, threadsPerBlock>>>(4, b, N);
  initWith<<<numberOfBlocks*N/threadsPerBlock+1, threadsPerBlock>>>(0, c, N);

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
  cudaDeviceSynchronize();

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
  cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}