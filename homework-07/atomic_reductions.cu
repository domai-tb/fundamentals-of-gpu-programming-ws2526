#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

const size_t VEC_SIZE = 40 * 1024 * 1024 / sizeof(float); // ~40MB vector

// Number of repetitions for timing
const unsigned int NUM_REPETITIONS_CPU  = 5;
const unsigned int NUM_REPETITIONS_GPU  = 50;

float EXEC_TIME_CPU        = 0.0f;
float EXEC_TIME_GPU_ATOMIC = 0.0f;
float EXEC_TIME_GPU_CASCADE= 0.0f;

// CUDA error check macro
void cudaErr(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/// CPU reduction
float reduceCPU(const float* h_vec, size_t size) {
    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    float sum = 0.0f;
    for (unsigned int r = 0; r < NUM_REPETITIONS_CPU; ++r) {
        float tmp = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            tmp += h_vec[i];
        }
        sum = tmp;
    }

    sdkStopTimer(&timer);
    EXEC_TIME_CPU = sdkGetTimerValue(&timer) / (float)NUM_REPETITIONS_CPU;
    return sum;
}

/// GPU reduction (global atomic)
__global__ void reduceAtomicGlobal(const float* d_vec, float* d_out, size_t size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    atomicAdd(d_out, d_vec[idx]);
}

/// GPU reduction (cascaded): each thread block reduces into shared memory
__global__ void reduceCascaded(const float* d_vec, float* d_out, size_t size) {

    extern __shared__ float shmem[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float local_sum = 0.0f;
    // Each thread processes two elements (coarsening)
    if (gid < size) local_sum += d_vec[gid];
    if (gid + blockDim.x < size) local_sum += d_vec[gid + blockDim.x];

    shmem[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmem[tid] += shmem[tid + s];
        }
        __syncthreads();
    }

    // thread 0 does global atomic add
    if (tid == 0) {
        atomicAdd(d_out, shmem[0]);
    }
}

void compareResults(const char* name, float ref, float gpu) {
    if (fabs(ref - gpu) > 1e-3f) {
        printf("%s: WARNING: mismatch (CPU=%.6f, GPU=%.6f)\n", name, ref, gpu);
    } else {
        printf("%s: result matches CPU reference (%.6f)\n", name, gpu);
    }
}

int main(int argc, char* argv[]) {
    srand(0);
    cudaErr(cudaSetDevice(0));

    // Host input
    float* h_vec  = (float*)malloc(VEC_SIZE * sizeof(float));
    if (!h_vec) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize vector
    for (size_t i = 0; i < VEC_SIZE; ++i) {
        h_vec[i] = (float)(rand() & 0xFFFF) / 65535.0f;
    }

    // CPU reduction
    printf("Running CPU reduction...\n");
    float cpuSum = reduceCPU(h_vec, VEC_SIZE);
    printf("CPU average time: %.3f ms\n", EXEC_TIME_CPU);

    // Device buffers
    float *d_vec = nullptr, *d_out = nullptr;
    cudaErr(cudaMalloc((void**)&d_vec, VEC_SIZE * sizeof(float)));
    cudaErr(cudaMalloc((void**)&d_out, sizeof(float)));

    cudaErr(cudaMemcpy(d_vec, h_vec, VEC_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // GPU: Atomic global reduction
    {
        // Zero output
        float zero = 0.0f;
        cudaErr(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));

        // Kernel launch parameters
        int threads = 256;
        int blocks  = (VEC_SIZE + threads - 1) / threads;

        // Warm-up
        reduceAtomicGlobal<<<blocks, threads>>>(d_vec, d_out, VEC_SIZE);
        cudaErr(cudaGetLastError());
        cudaErr(cudaDeviceSynchronize());

        StopWatchInterface* timer = NULL;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        for (unsigned int r = 0; r < NUM_REPETITIONS_GPU; ++r) {
            cudaErr(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));
            reduceAtomicGlobal<<<blocks, threads>>>(d_vec, d_out, VEC_SIZE);
            cudaErr(cudaGetLastError());
            cudaErr(cudaDeviceSynchronize());
        }

        sdkStopTimer(&timer);
        EXEC_TIME_GPU_ATOMIC = sdkGetTimerValue(&timer) / (float)NUM_REPETITIONS_GPU;

        float gpuAtomicSum = 0.0f;
        cudaErr(cudaMemcpy(&gpuAtomicSum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("GPU atomic reduction average time: %.3f ms\n", EXEC_TIME_GPU_ATOMIC);
        compareResults("Atomic global GPU", cpuSum, gpuAtomicSum);
    }

    // GPU: Cascaded reduction
    {
        // Zero output
        float zero = 0.0f;
        cudaErr(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (VEC_SIZE + (threads * 2 - 1)) / (threads * 2);
        size_t sharedBytes = threads * sizeof(float);

        // Warm-up
        reduceCascaded<<<blocks, threads, sharedBytes>>>(d_vec, d_out, VEC_SIZE);
        cudaErr(cudaGetLastError());
        cudaErr(cudaDeviceSynchronize());

        StopWatchInterface* timer = NULL;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        for (unsigned int r = 0; r < NUM_REPETITIONS_GPU; ++r) {
            cudaErr(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));
            reduceCascaded<<<blocks, threads, sharedBytes>>>(d_vec, d_out, VEC_SIZE);
            cudaErr(cudaGetLastError());
            cudaErr(cudaDeviceSynchronize());
        }

        sdkStopTimer(&timer);
        EXEC_TIME_GPU_CASCADE = sdkGetTimerValue(&timer) / (float)NUM_REPETITIONS_GPU;

        float gpuCascadeSum = 0.0f;
        cudaErr(cudaMemcpy(&gpuCascadeSum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("GPU cascaded reduction average time: %.3f ms\n", EXEC_TIME_GPU_CASCADE);
        compareResults("Cascaded GPU", cpuSum, gpuCascadeSum);
    }

    // Clean up
    cudaErr(cudaFree(d_vec));
    cudaErr(cudaFree(d_out));
    free(h_vec);

    cudaErr(cudaDeviceReset());
    return EXIT_SUCCESS;
}