#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

using std::vector;
using namespace std::chrono;

#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline int minVal(int x, int y)
{
	return (x < y) ? x : y;
}

constexpr int threads_per_block = 256;
constexpr int N = 1024;

__global__ void dotproduct_gpu(float* out, float* x, float* y, int size)
{
	__shared__ float cache[threads_per_block];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float temp = 0.0;
	while (idx < size)
	{
		temp += x[idx] * y[idx];
		idx += (gridDim.x * blockDim.x);
	}

	cache[threadIdx.x] = temp;
	__syncthreads();

	int i = threads_per_block / 2;
	while (i > 0)
	{
		if (threadIdx.x < i) {
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		out[blockIdx.x] = cache[0];
	}
}

float dotproduct(vector<float> a, vector<float> b)
{
	size_t N = a.size();

	float* x;
	float* y;
	float* out;

	int blockNum = minVal(32, ((N + threads_per_block - 1) / threads_per_block));

	check(cudaMalloc((void**)&x, N * sizeof(float)));
	check(cudaMalloc((void**)&y, N * sizeof(float)));
	check(cudaMalloc((void**)&out, sizeof(float) * blockNum));

	check(cudaMemcpy(x, a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(y, b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

	dotproduct_gpu << < blockNum, threads_per_block>> > (out, x, y, N);

	float* retPointer = (float*)malloc(sizeof(float) * blockNum);
	check(cudaMemcpy(retPointer, out, sizeof(float) * blockNum, cudaMemcpyDeviceToHost));

	cudaFree(x);
	cudaFree(y);
	cudaFree(out);

	float sum = 0.0;
	for (int i = 0; i < blockNum; i++)
	{
		sum += retPointer[i];
	}

	free(retPointer);

	return sum;
}
int main(void)
{
	vector<float> x(100000, 2);
	vector<float> y(100000, 1);

	y[400] = 4;
	x[400] = 12;

	float dot = dotproduct(x, y);

	std::cout << "Dot product: " << dot << "\n";
	return 0;
}