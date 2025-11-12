#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define NUM_ITERATIONS ( 1 << 15 )

const int TEST_NUM = 11;
const int TEST_THREAD_NUM[TEST_NUM] = {1,2,4,8,16,32,64,128,256,512,1024};

float *var1, *var2, *var3, *var4; // global variables for memory test

// Kernel for 1-time ILP
__global__ void ilp1Kernel(float *a_out) {
    float a = 1.0, b = 2.0, c = 3.0;
    #pragma unroll 16
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        a = a * b + c;
    }
    a_out[threadIdx.x] = a;
}

// Kernel for 4-time ILP
__global__ void ilp4Kernel(float *a_out, float *d_out, float *e_out, float *f_out) {
    float a = 1.0, b = 2.0, c = 3.0, d = 4.0, e = 3.0, f = 2.0;
    #pragma unroll 16
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        a = a * b + c;
        d = d * b + c;
        e = e * b + c;
        f = f * b + c;
    }
    a_out[threadIdx.x] = a;
    d_out[threadIdx.x] = d;
    e_out[threadIdx.x] = e;
    f_out[threadIdx.x] = f;
}

// Error checking function
void cudaErr(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void testWithThreadNum(unsigned int numThreadsPerBlock) {
    int numOperations;

    // Allocate memory for dummy arrays
    cudaErr(cudaMalloc((void**)&var1, numThreadsPerBlock * sizeof(float)));
    cudaErr(cudaMalloc((void**)&var2, numThreadsPerBlock * sizeof(float)));
    cudaErr(cudaMalloc((void**)&var3, numThreadsPerBlock * sizeof(float)));
    cudaErr(cudaMalloc((void**)&var4, numThreadsPerBlock * sizeof(float)));
    
    // Start timer
	float tStart, tEnd;
	StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
	tStart = sdkGetTimerValue(&timer);
	sdkStartTimer(&timer);

    ilp1Kernel<<<1, numThreadsPerBlock>>>(var1);
    cudaErr(cudaDeviceSynchronize());

    // Stop timer
    sdkStopTimer(&timer);
	tEnd = sdkGetTimerValue(&timer);

    numOperations = NUM_ITERATIONS * numThreadsPerBlock * 2;
    printf("[ ILP1 ] Threads = %d | Execution time per operation: %ems\n", numThreadsPerBlock, (tEnd - tStart) * 1000 / numOperations);

    // Start timer
	tStart = sdkGetTimerValue(&timer);
	sdkStartTimer(&timer);

    ilp4Kernel<<<1, numThreadsPerBlock>>>(var1, var2, var3, var4);
    cudaErr(cudaDeviceSynchronize());

    // Stop timer
    sdkStopTimer(&timer);
	tEnd = sdkGetTimerValue(&timer);
    
    numOperations = NUM_ITERATIONS * numThreadsPerBlock * 2 * 4;
    printf("[ ILP4 ] Threads = %d | Execution time per operation: %ems\n", numThreadsPerBlock, (tEnd - tStart) * 1000 / numOperations);

    // Clean up
    cudaErr(cudaFree(var1)); 
    cudaErr(cudaFree(var2)); 
    cudaErr(cudaFree(var3)); 
    cudaErr(cudaFree(var4));
}

int main(int argc, char **argv) {
    int numThreadsPerBlock;

    cudaErr(cudaSetDevice(0));
    
    // Read command line arguments
    if (argc > 2) {
        printf("Usage: %s <num threads>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (argc != 2) {
        // Just run test with array of thread numbers
        for (int i = 0; i < TEST_NUM; i++) {
            numThreadsPerBlock = TEST_THREAD_NUM[i];
            testWithThreadNum(numThreadsPerBlock);
        }
    } else {
        // Use provided thread number
        numThreadsPerBlock = atoi(argv[1]);
        if (numThreadsPerBlock > 1024 || numThreadsPerBlock <= 0) {
            printf("Number of threads must be in the range of 0 to 1024!\n");
            exit(EXIT_FAILURE);
        }

        testWithThreadNum(numThreadsPerBlock);
    }
    
    cudaErr(cudaDeviceReset());

    return EXIT_SUCCESS;
}
