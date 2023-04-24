#include <stdio.h>
#include <unistd.h>

__global__ void printNumber(int number)
{
  printf("%d\n", number);
}

int main()
{
  for (int i = 0; i < 5; ++i)
  {
    printNumber<<<1, 1>>>(i);
  }
  cudaDeviceSynchronize();
}

