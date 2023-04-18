#include <stdio.h>

inline cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess){
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(result));
    }
    return result;
}

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i+=stride){
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
    const int N = 2<<20;
    size_t size = N * sizeof(float);

    float *a;
    float *b;
    float *c;

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);
    
    int threads_per_block = 1024;
    int number_of_blocks = N < 1024*32 ? (N+threads_per_block-1)/threads_per_block : 32;

    addVectorsInto<<<number_of_blocks, threads_per_block>>>(c, a, b, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Error: %s", cudaGetErrorString(err));
    }
    else printf("Success!");
    checkCuda(cudaDeviceSynchronize());
    checkElementsAre(7, c, N);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
