#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

__global__ void helloFromGPU(void) {
    printf("Hello World from GPU!\n");
}

int main(void) {
    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();

    return 0;
}