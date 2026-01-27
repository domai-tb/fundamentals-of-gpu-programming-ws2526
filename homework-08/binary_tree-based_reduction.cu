#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

/// CPU reduction (reference)
float reduceCPU(const float* h_vec, size_t size) {
    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    float sum = 0.0f;
    for (unsigned int r = 0; r < NUM_REPETITIONS_CPU; ++r) {
        double tmp = 0.0;
        for (size_t i = 0; i < size; ++i) {
            tmp += (double)h_vec[i];
        }
        sum = (float)tmp;
    }

    sdkStopTimer(&timer);
    EXEC_TIME_CPU = sdkGetTimerValue(&timer) / (float)NUM_REPETITIONS_CPU;
    return sum;
}

/// Reduction in shared memory (Harris-style unrolling)
__device__ __forceinline__ void reduceBlock(volatile float* sdata, unsigned int tid, unsigned int blockSize) {
    if (blockSize >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }

    // Unroll last warp (no __syncthreads needed)
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid +  8];
        sdata[tid] += sdata[tid +  4];
        sdata[tid] += sdata[tid +  2];
        sdata[tid] += sdata[tid +  1];
    }
}

/// GPU reduction (atomic, optimized): fixed blocks, grid-stride, one atomicAdd per block
__global__ void reduceCascadedAtomic(const float* d_vec, float* d_out, size_t size) {

    extern __shared__ float shmem[];
    unsigned int tid = threadIdx.x;

    unsigned int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x * 2;

    float local_sum = 0.0f;

    // Grid-stride loop, coarsening (2 elements per iteration)
    while (gid < size) {
        local_sum += d_vec[gid];
        if (gid + blockDim.x < size) local_sum += d_vec[gid + blockDim.x];
        gid += stride;
    }

    shmem[tid] = local_sum;
    __syncthreads();

    reduceBlock((volatile float*)shmem, tid, blockDim.x);

    if (tid == 0) {
        atomicAdd(d_out, shmem[0]);
    }
}

/// GPU reduction (cascaded single-pass): fixed blocks + __threadfence final stage
__global__ void reduceCascadedThreadfence(const float* d_vec, float* d_out, float* d_partials, unsigned int* d_count, size_t size) {

    extern __shared__ float shmem[];
    __shared__ bool is_last_block;

    unsigned int tid = threadIdx.x;

    unsigned int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x * 2;

    float local_sum = 0.0f;

    // Grid-stride loop, coarsening (2 elements per iteration)
    while (gid < size) {
        local_sum += d_vec[gid];
        if (gid + blockDim.x < size) local_sum += d_vec[gid + blockDim.x];
        gid += stride;
    }

    shmem[tid] = local_sum;
    __syncthreads();

    reduceBlock((volatile float*)shmem, tid, blockDim.x);

    if (tid == 0) {
        d_partials[blockIdx.x] = shmem[0];

        // Ensure partials are visible before we signal completion
        __threadfence();

        unsigned int ticket = atomicInc(d_count, (unsigned int)(gridDim.x - 1));
        is_last_block = (ticket == (unsigned int)(gridDim.x - 1));
    }
    __syncthreads();

    // Last block finishes the reduction
    if (is_last_block) {
        float sum = 0.0f;

        // Each thread accumulates multiple partials
        for (unsigned int i = tid; i < (unsigned int)gridDim.x; i += blockDim.x) {
            sum += d_partials[i];
        }

        shmem[tid] = sum;
        __syncthreads();

        reduceBlock((volatile float*)shmem, tid, blockDim.x);

        if (tid == 0) {
            *d_out = shmem[0];
            *d_count = 0;
        }
    }
}

void compareResults(const char* name, float ref, float gpu) {
    float diff = (float)fabs((double)ref - (double)gpu);
    float denom = (float)fabs((double)ref);
    float rel = diff / (denom + 1e-6f);

    if (rel > 1e-4f && diff > 1e-2f) {
        printf("%s: WARNING: mismatch (CPU=%.6f, GPU=%.6f, diff=%.6f, rel=%.6e)\n", name, ref, gpu, diff, rel);
    } else {
        printf("%s: result matches CPU reference (%.6f)\n", name, gpu);
    }
}

/// Compute fixed number of blocks ~= active blocks on GPU (occupancy-based)
int getFixedBlocks(int threads, size_t sharedBytes) {
    cudaDeviceProp prop;
    cudaErr(cudaGetDeviceProperties(&prop, 0));

    int blocksPerSM = 0;
    cudaErr(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, reduceCascadedThreadfence, threads, sharedBytes));

    int blocks = blocksPerSM * prop.multiProcessorCount;
    if (blocks < 1) blocks = 1;

    // Conservative cap for older grid dimension limits
    if (blocks > 65535) blocks = 65535;

    return blocks;
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
    float *d_vec = nullptr, *d_out = nullptr, *d_partials = nullptr;
    unsigned int *d_count = nullptr;

    cudaErr(cudaMalloc((void**)&d_vec, VEC_SIZE * sizeof(float)));
    cudaErr(cudaMalloc((void**)&d_out, sizeof(float)));

    cudaErr(cudaMemcpy(d_vec, h_vec, VEC_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Fixed launch configuration (approx. active blocks)
    int threads = 256;
    size_t sharedBytes = threads * sizeof(float);
    int blocks = getFixedBlocks(threads, sharedBytes);

    cudaErr(cudaMalloc((void**)&d_partials, blocks * sizeof(float)));
    cudaErr(cudaMalloc((void**)&d_count, sizeof(unsigned int)));

    printf("Using fixed configuration: blocks=%d, threads=%d\n", blocks, threads);

    // GPU: Fast atomic reduction kernel (baseline from previous homework)
    {
        // Zero output
        float zero = 0.0f;
        unsigned int zcount = 0;
        cudaErr(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));
        cudaErr(cudaMemcpy(d_count, &zcount, sizeof(unsigned int), cudaMemcpyHostToDevice));

        // Warm-up
        reduceCascadedAtomic<<<blocks, threads, sharedBytes>>>(d_vec, d_out, VEC_SIZE);
        cudaErr(cudaGetLastError());
        cudaErr(cudaDeviceSynchronize());

        StopWatchInterface* timer = NULL;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        for (unsigned int r = 0; r < NUM_REPETITIONS_GPU; ++r) {
            cudaErr(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));
            reduceCascadedAtomic<<<blocks, threads, sharedBytes>>>(d_vec, d_out, VEC_SIZE);
            cudaErr(cudaGetLastError());
            cudaErr(cudaDeviceSynchronize());
        }

        sdkStopTimer(&timer);
        EXEC_TIME_GPU_ATOMIC = sdkGetTimerValue(&timer) / (float)NUM_REPETITIONS_GPU;

        float gpuAtomicSum = 0.0f;
        cudaErr(cudaMemcpy(&gpuAtomicSum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("GPU atomic (optimized) reduction average time: %.3f ms\n", EXEC_TIME_GPU_ATOMIC);
        compareResults("Atomic optimized GPU", cpuSum, gpuAtomicSum);
    }

    // GPU: Cascaded single-pass reduction with __threadfence final stage (HW#8)
    {
        // Zero output and counter
        float zero = 0.0f;
        unsigned int zcount = 0;
        cudaErr(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));
        cudaErr(cudaMemcpy(d_count, &zcount, sizeof(unsigned int), cudaMemcpyHostToDevice));

        // Warm-up
        reduceCascadedThreadfence<<<blocks, threads, sharedBytes>>>(d_vec, d_out, d_partials, d_count, VEC_SIZE);
        cudaErr(cudaGetLastError());
        cudaErr(cudaDeviceSynchronize());

        StopWatchInterface* timer = NULL;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        for (unsigned int r = 0; r < NUM_REPETITIONS_GPU; ++r) {
            cudaErr(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));
            cudaErr(cudaMemcpy(d_count, &zcount, sizeof(unsigned int), cudaMemcpyHostToDevice));
            reduceCascadedThreadfence<<<blocks, threads, sharedBytes>>>(d_vec, d_out, d_partials, d_count, VEC_SIZE);
            cudaErr(cudaGetLastError());
            cudaErr(cudaDeviceSynchronize());
        }

        sdkStopTimer(&timer);
        EXEC_TIME_GPU_CASCADE = sdkGetTimerValue(&timer) / (float)NUM_REPETITIONS_GPU;

        float gpuCascadeSum = 0.0f;
        cudaErr(cudaMemcpy(&gpuCascadeSum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("GPU cascaded (__threadfence) reduction average time: %.3f ms\n", EXEC_TIME_GPU_CASCADE);
        compareResults("Cascaded threadfence GPU", cpuSum, gpuCascadeSum);
    }

    // Summary
    printf("\nSummary:\n");
    printf("CPU average time: %.3f ms\n", EXEC_TIME_CPU);
    printf("GPU atomic optimized average time: %.3f ms\n", EXEC_TIME_GPU_ATOMIC);
    printf("GPU cascaded threadfence average time: %.3f ms\n", EXEC_TIME_GPU_CASCADE);

    // Clean up
    cudaErr(cudaFree(d_vec));
    cudaErr(cudaFree(d_out));
    cudaErr(cudaFree(d_partials));
    cudaErr(cudaFree(d_count));
    free(h_vec);

    cudaErr(cudaDeviceReset());
    return EXIT_SUCCESS;
}
