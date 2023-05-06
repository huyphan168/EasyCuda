#include <torch/extension.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_DIM(A, B) AT_ASSERTM((A.dim() <= 2 || B.dim() <= 2), #A "Assertion Failed, A or B tensor has dimensionality larger than 2 which is not supported")


torch::Tensor parallelMatMul_cuda(torch::Tensor A, torch::Tensor B, int T, int TB);

torch::Tensor parallelMatMul(torch::Tensor A, torch::Tensor B, int T, int TB) {
	CHECK_DIM(A,B);
	return parallelMatMul_cuda(A, B, T, TB);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parallelMatMul", &parallelMatMul, "Parallel matrix multiplication (CUDA)");
}
