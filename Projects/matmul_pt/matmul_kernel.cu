#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (index < M){
        for (int i = index; i < M; i+=stride){
            for (int k = 0; k < K; ++k){
                for (int j = 0; j < N; ++j){
                    C[i*K + k] += A[i*N + j] * B[j*K + k];
                }
            }
        }
    } 
}

__global__ void matrix_vector_multiply_kernel(const float* A, const float* B, float*C, int M, int N){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
	if (index < M){
		for (int i = index; i < M; i+=stride){
            for (int j = 0; j < N; j++){
                C[i] += A[i*N + j] * B[j];
            }
        }
	}
}

__global__ void dot_product_kernel(const float* A, const float* B, float* C, int K){
  	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	if (index == 0){
		for (int i = 0; i<K; i++){
			sum += A[i] * B[i]; 
		}
		C[0] = sum;
	}
}

torch::Tensor parallelMatMul_cuda(torch::Tensor A, torch::Tensor B, int T, int TB){
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	if (A.dim() == 1 && B.dim() == 1){
		int K = A.size(0);
		auto C = torch::zeros({1}, options);
		dot_product_kernel<<<TB, T>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K);
		cudaDeviceSynchronize();
        return C;
	}
	if (A.dim() == 2 && B.dim() == 2){
		int M = A.size(-2);
		int N = A.size(-1);
		int K = B.size(-1);
		auto C = torch::zeros({M, K}, options);
		matrix_multiply_kernel<<<TB, T>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
        cudaDeviceSynchronize();
        return C;
	}
	if (A.dim() == 1 && B.dim() == 2){
		A.unsqueeze(0);
		int M = A.size(-2);
		int N = A.size(-1);
		int K = B.size(-1);
		auto C = torch::zeros({M, K}, options);
		matrix_multiply_kernel<<<TB, T>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
        cudaDeviceSynchronize();
        return C;
	}
	if (A.dim() == 2 && B.dim() == 1){
		int M = A.size(-2);
		int N = A.size(-1);
        auto C = torch::zeros({M}, options);
		matrix_vector_multiply_kernel<<<TB, T>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N);
        cudaDeviceSynchronize();
        return C;
	}
	cudaDeviceSynchronize();
}

