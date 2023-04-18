#include<stdio.h>
#include<assert.h>

inline cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess){
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(result));
    }
    return result;
}

int main(void){
    checkCuda(cudaDeviceSynchronize());
}