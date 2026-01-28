#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

/// Scan configuration
const unsigned int SCAN_BLOCK_SIZE        = 512;                 // threads per block
const unsigned int ELEMENTS_PER_BLOCK     = 2 * SCAN_BLOCK_SIZE; // 1024 elems per block segment
const unsigned int MAX_SMALL_SCAN_ELEMS   = 1024;                // for scanning block sums (<=1024)

/// Random input generation
const int RAND_MIN_VALUE = 0;
const int RAND_MAX_VALUE = 10;  // keep sums safely in 32-bit for n <= 1e6

/// Number of repetitions for timing
const unsigned int NUM_REPETITIONS_CPU = 20;
const unsigned int NUM_REPETITIONS_GPU = 200;

/// Global execution time accumulators (ms)
float EXECUTION_TIME_CPU = 0.0f;
float EXECUTION_TIME_GPU = 0.0f;

/// CUDA error checking helper
void cudaErr(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/// 1) CPU inclusive scan (prefix sum), O(n)
void scanCPU(const int* in, int* out, unsigned int n) {
    long long sum = 0;
    for (unsigned int i = 0; i < n; ++i) {
        sum += (long long)in[i];
        out[i] = (int)sum;
    }
}

// 2) GPU kernels

/// Brent-Kung Kernel Block
__global__ void scanBrentKungBlockKernel(const int* d_in, int* d_out, int* d_blockSums, unsigned int n) {
    __shared__ int s_data[ELEMENTS_PER_BLOCK];

    unsigned int tid  = threadIdx.x;
    unsigned int base = blockIdx.x * ELEMENTS_PER_BLOCK;

    unsigned int i0 = base + (2 * tid);
    unsigned int i1 = i0 + 1;

    int x0 = 0;
    int x1 = 0;

    if (i0 < n) x0 = d_in[i0];
    if (i1 < n) x1 = d_in[i1];

    s_data[2 * tid]     = x0;
    s_data[2 * tid + 1] = x1;

    __syncthreads();

    // Reduction phase
    for (unsigned int stride = 1; stride < ELEMENTS_PER_BLOCK; stride <<= 1) {
        unsigned int idx = (tid + 1) * stride * 2 - 1;
        if (idx < ELEMENTS_PER_BLOCK) {
            s_data[idx] += s_data[idx - stride];
        }
        __syncthreads();
    }

    // Total sum of this block segment (zeros padded for out-of-range)
    int blockTotal = s_data[ELEMENTS_PER_BLOCK - 1];

    // Set last element to 0 for exclusive down-sweep
    if (tid == 0) {
        s_data[ELEMENTS_PER_BLOCK - 1] = 0;
    }
    __syncthreads();

    // Distribution phase
    for (unsigned int stride = ELEMENTS_PER_BLOCK >> 1; stride >= 1; stride >>= 1) {
        unsigned int idx = (tid + 1) * stride * 2 - 1;
        if (idx < ELEMENTS_PER_BLOCK) {
            int t = s_data[idx - stride];
            s_data[idx - stride] = s_data[idx];
            s_data[idx] += t;
        }
        __syncthreads();

        if (stride == 1) {
            break; // prevent unsigned underflow
        }
    }

    // Exclusive -> Inclusive by adding original values
    if (i0 < n) d_out[i0] = s_data[2 * tid] + x0;
    if (i1 < n) d_out[i1] = s_data[2 * tid + 1] + x1;

    // Store per-block total sum (for multi-block scan)
    if (d_blockSums != NULL && tid == 0) {
        d_blockSums[blockIdx.x] = blockTotal;
    }
}

/// Brent-Kung Kernel Small
__global__ void scanBrentKungSmallKernel(const int* d_in, int* d_out, unsigned int n) {
    __shared__ int s_data[MAX_SMALL_SCAN_ELEMS];

    unsigned int tid = threadIdx.x;

    unsigned int i0 = 2 * tid;
    unsigned int i1 = i0 + 1;

    int x0 = 0;
    int x1 = 0;

    if (i0 < n) x0 = d_in[i0];
    if (i1 < n) x1 = d_in[i1];

    s_data[i0] = x0;
    s_data[i1] = x1;

    __syncthreads();

    // Up-sweep
    for (unsigned int stride = 1; stride < MAX_SMALL_SCAN_ELEMS; stride <<= 1) {
        unsigned int idx = (tid + 1) * stride * 2 - 1;
        if (idx < MAX_SMALL_SCAN_ELEMS) {
            s_data[idx] += s_data[idx - stride];
        }
        __syncthreads();
    }

    // Exclusive setup
    if (tid == 0) {
        s_data[MAX_SMALL_SCAN_ELEMS - 1] = 0;
    }
    __syncthreads();

    // Down-sweep
    for (unsigned int stride = MAX_SMALL_SCAN_ELEMS >> 1; stride >= 1; stride >>= 1) {
        unsigned int idx = (tid + 1) * stride * 2 - 1;
        if (idx < MAX_SMALL_SCAN_ELEMS) {
            int t = s_data[idx - stride];
            s_data[idx - stride] = s_data[idx];
            s_data[idx] += t;
        }
        __syncthreads();

        if (stride == 1) {
            break;
        }
    }

    // Exclusive -> inclusive (store only valid outputs)
    if (i0 < n) d_out[i0] = s_data[i0] + x0;
    if (i1 < n) d_out[i1] = s_data[i1] + x1;
}

/// Add scanned block offsets to each element of the per-block scanned output.
/// For inclusive scan, block i adds prefix sum of blocks [0..i-1].
__global__ void addBlockOffsetsKernel(int* d_out, const int* d_scannedBlockSums, unsigned int n) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }

    unsigned int blockId = gid / ELEMENTS_PER_BLOCK;
    if (blockId == 0) {
        return;
    }

    int offset = d_scannedBlockSums[blockId - 1];
    d_out[gid] += offset;
}

/// 3) CPU/GPU output comparison
bool compareResults(const int* ref, const int* gpu, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        if (ref[i] != gpu[i]) {
            printf("Mismatch at i=%u: CPU=%d GPU=%d\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

/// 4) Additional helper functions
void generateRandomIntVector(int* data, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        int r = rand();
        data[i] = RAND_MIN_VALUE + (r % (RAND_MAX_VALUE - RAND_MIN_VALUE + 1));
    }
}

/// Measure CPU scan average time (ms)
void cpuScanTimed(const int* in, int* out, unsigned int n) {
    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);

    sdkStartTimer(&timer);
    float tStart = sdkGetTimerValue(&timer);

    for (unsigned int r = 0; r < NUM_REPETITIONS_CPU; ++r) {
        scanCPU(in, out, n);
    }

    sdkStopTimer(&timer);
    float tEnd = sdkGetTimerValue(&timer);

    EXECUTION_TIME_CPU = (tEnd - tStart) / (float)NUM_REPETITIONS_CPU;
}

/// Measure GPU scan average time (ms) over NUM_REPETITIONS_GPU
int gpuScanTimed(const int* hIn, int* hOut, unsigned int n) {
    int* dIn  = NULL;
    int* dOut = NULL;

    size_t bytes = (size_t)n * sizeof(int);
    cudaErr(cudaMalloc((void**)&dIn, bytes));
    cudaErr(cudaMalloc((void**)&dOut, bytes));

    // H2D copy not included in GPU timing
    cudaErr(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    unsigned int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    int* dBlockSums        = NULL;
    int* dScannedBlockSums = NULL;

    if (numBlocks > 1) {
        cudaErr(cudaMalloc((void**)&dBlockSums, (size_t)numBlocks * sizeof(int)));
        cudaErr(cudaMalloc((void**)&dScannedBlockSums, (size_t)numBlocks * sizeof(int)));
    }

    dim3 blockScan(SCAN_BLOCK_SIZE);
    dim3 gridScan(numBlocks);

    dim3 blockAdd(256);
    dim3 gridAdd((n + blockAdd.x - 1) / blockAdd.x);

    // Warm-up
    scanBrentKungBlockKernel<<<gridScan, blockScan>>>(dIn, dOut, dBlockSums, n);
    cudaErr(cudaGetLastError());

    if (numBlocks > 1) {
        scanBrentKungSmallKernel<<<1, blockScan>>>(dBlockSums, dScannedBlockSums, numBlocks);
        cudaErr(cudaGetLastError());

        addBlockOffsetsKernel<<<gridAdd, blockAdd>>>(dOut, dScannedBlockSums, n);
        cudaErr(cudaGetLastError());
    }

    cudaErr(cudaDeviceSynchronize());

    // Timed repetitions: CUDA events measure kernel time
    cudaEvent_t start, stop;
    cudaErr(cudaEventCreate(&start));
    cudaErr(cudaEventCreate(&stop));

    cudaErr(cudaEventRecord(start, 0));

    for (unsigned int r = 0; r < NUM_REPETITIONS_GPU; ++r) {
        scanBrentKungBlockKernel<<<gridScan, blockScan>>>(dIn, dOut, dBlockSums, n);
        cudaErr(cudaGetLastError());

        if (numBlocks > 1) {
            scanBrentKungSmallKernel<<<1, blockScan>>>(dBlockSums, dScannedBlockSums, numBlocks);
            cudaErr(cudaGetLastError());

            addBlockOffsetsKernel<<<gridAdd, blockAdd>>>(dOut, dScannedBlockSums, n);
            cudaErr(cudaGetLastError());
        }
    }

    cudaErr(cudaEventRecord(stop, 0));
    cudaErr(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    cudaErr(cudaEventElapsedTime(&totalMs, start, stop));
    EXECUTION_TIME_GPU = totalMs / (float)NUM_REPETITIONS_GPU;

    cudaErr(cudaEventDestroy(start));
    cudaErr(cudaEventDestroy(stop));

    // D2H copy not included in GPU timing
    cudaErr(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));

    if (dBlockSums != NULL) cudaErr(cudaFree(dBlockSums));
    if (dScannedBlockSums != NULL) cudaErr(cudaFree(dScannedBlockSums));
    cudaErr(cudaFree(dIn));
    cudaErr(cudaFree(dOut));

    return EXIT_SUCCESS;
}

/// Main 
int main(int argc, char* argv[]) {
    srand(0);

    int deviceId = 0;
    cudaErr(cudaSetDevice(deviceId));

    printf("Block scan: %u threads -> %u elements per block\n", SCAN_BLOCK_SIZE, ELEMENTS_PER_BLOCK);
    printf("Timing repetitions: CPU=%u, GPU=%u\n\n", NUM_REPETITIONS_CPU, NUM_REPETITIONS_GPU);

    printf("|   n   | CPU time [ms] | GPU time [ms] | Speedup |\n");
    printf("|-------|---------------|---------------|---------|\n");

    for (unsigned int n = 100000; n <= 1000000; n += 100000) {
        int* hIn  = (int*)malloc((size_t)n * sizeof(int));
        int* hCPU = (int*)malloc((size_t)n * sizeof(int));
        int* hGPU = (int*)malloc((size_t)n * sizeof(int));

        if (hIn == NULL || hCPU == NULL || hGPU == NULL) {
            fprintf(stderr, "Fatal: failed to allocate host arrays for n=%u.\n", n);
            return EXIT_FAILURE;
        }

        generateRandomIntVector(hIn, n);

        // CPU timing
        cpuScanTimed(hIn, hCPU, n);

        // GPU timing + correctness output
        if (gpuScanTimed(hIn, hGPU, n) != EXIT_SUCCESS) {
            fprintf(stderr, "GPU scan failed for n=%u.\n", n);
            return EXIT_FAILURE;
        }

        bool ok = compareResults(hCPU, hGPU, n);
        if (!ok) {
            printf("WARNING: result differs from CPU reference for n=%u!\n", n);
        }

        float speedup = (EXECUTION_TIME_GPU > 0.0f) ? (EXECUTION_TIME_CPU / EXECUTION_TIME_GPU) : 0.0f;
        printf("| %5u | %13.3f | %13.3f | %7.2fx |\n", n, EXECUTION_TIME_CPU, EXECUTION_TIME_GPU, speedup);

        free(hIn);
        free(hCPU);
        free(hGPU);
    }

    cudaErr(cudaDeviceReset());
    return EXIT_SUCCESS;
}
