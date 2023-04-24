#include <stdio.h>

__global__
void initWith(float num, float *a, int N)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    a[i] = num;
  }
}

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
  int deviceId;
  int numberOfSMs;
  const int numberOfStreams = 8;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  const int N = 2<<24;
  const int streamN = N / numberOfStreams;

  size_t streamSize = streamN * sizeof(float);

  float *a[numberOfStreams];
  float *b[numberOfStreams];
  float *c[numberOfStreams];
  float *h_c[numberOfStreams];

  cudaStream_t streams[numberOfStreams];
  cudaStream_t memoryStream;

  cudaStreamCreate(&memoryStream);

  for (int i = 0; i < numberOfStreams; ++i)
  {
    cudaStreamCreate(&streams[i]);
    cudaMalloc(&a[i], streamSize);
    cudaMalloc(&b[i], streamSize);
    cudaMalloc(&c[i], streamSize);
    h_c[i] = (float *)malloc(streamSize);
  }

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  for (int i = 0; i < numberOfStreams; ++i)
  {
    initWith<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(3, a[i], streamN);
    initWith<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(4, b[i], streamN);
    initWith<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(0, c[i], streamN);

    addVectorsInto<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(c[i], a[i], b[i], streamN);

    cudaMemcpyAsync(h_c[i], c[i], streamSize, cudaMemcpyDeviceToHost, streams[i]);
    // cudaMemcpyAsync(h_c[i], c[i], streamSize, cudaMemcpyDeviceToHost, memoryStream);
  }

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));


  /*
   * Destroy streams when they are no longer needed.
   */

  for (int i = 0; i < numberOfStreams; ++i)
  {
    checkElementsAre(7, h_c[i], streamN);
    cudaStreamDestroy(streams[i]);
    cudaFree(a[i]);
    cudaFree(b[i]);
    cudaFree(c[i]);
    free(h_c[i]);
  }
  cudaStreamDestroy(memoryStream);

}
