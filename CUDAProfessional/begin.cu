#include<iostream>
#include<stdio.h>
#include<cuda_runtime.h>

using namespace std;

__global__ void helloFromGPU(void){
    if (threadIdx.x == 5)
        printf("Hello World from GPU! threaedId.x = %d", threadIdx.x);
};

void helloFromCPU(void){
    printf("Hello World from CPU!\n");
}

int main(void){

    helloFromCPU();
    helloFromGPU<<<1,10>>>();
    // cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;

}