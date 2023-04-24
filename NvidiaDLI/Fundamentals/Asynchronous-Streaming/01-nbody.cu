#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */

__global__
void bodyForce(Body *p, float dt, int n){
  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
  float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
  if (0 < index < n){
    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[index].x;
      float dy = p[j].y - p[index].y;
      float dz = p[j].z - p[index].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }
    p[index].vx += dt*Fx; p[index].vy += dt*Fy; p[index].vz += dt*Fz;
  }
}



__global__ 
void integration(Body *p, float dt, int nBodies){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < nBodies){
    p[index].x += p[index].vx*dt;
    p[index].y += p[index].vy*dt;
    p[index].z += p[index].vz*dt;
  }
}


int main(const int argc, const char** argv) {

  // The assessment will test against both 2<11 and 2<15.
  // Feel free to pass the command line argument 15 when you generate ./nbody report files
  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  // The assessment will pass hidden initialized values to check for correctness.
  // You should not make changes to these files, or else the assessment will not work.
  const char * initialized_values;
  const char * solution_values;

  if (nBodies == 2<<11) {
    initialized_values = "09-nbody/files/initialized_4096";
    solution_values = "09-nbody/files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "09-nbody/files/initialized_65536";
    solution_values = "09-nbody/files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;

  cudaMallocManaged(&buf, bytes);
  Body *p = (Body*)buf;

  read_values_from_file(initialized_values, buf, bytes);

  // Cuda variables
  int deviceId;
  cudaGetDevice(&deviceId);
  int numberOfSMs;
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  double totalTime = 0.0;
  cudaMemPrefetchAsync(p, bytes ,deviceId);

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

  /*
   * You will likely wish to refactor the work being done in `bodyForce`,
   * and potentially the work to integrate the positions.
   */
    unsigned int numberThreads = 256;
    unsigned int numberBlocks = ((nBodies / numberThreads) / numberOfSMs + 1)*numberOfSMs;

    bodyForce<<<numberBlocks, numberThreads>>>(p, dt, nBodies); // compute interbody forces
    cudaDeviceSynchronize();
  /*
   * This position integration cannot occur until this round of `bodyForce` has completed.
   * Also, the next round of `bodyForce` cannot begin until the integration is complete.
   */
    integration<<<numberBlocks, numberThreads>>>(p, dt, nBodies);
    cudaDeviceSynchronize();

    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }
  cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId);

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
  write_values_to_file(solution_values, buf, bytes);

  // You will likely enjoy watching this value grow as you accelerate the application,
  // but beware that a failure to correctly synchronize the device might result in
  // unrealistically high values.
  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

  cudaFree(buf);
}
