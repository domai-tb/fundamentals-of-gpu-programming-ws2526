#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_functions.h>

/// Matrix type definition
typedef struct {
    unsigned int dimX;
    unsigned int dimY;
    float* elements;
} Matrix;

/// Tile configuration for transpose / copy
const unsigned int TILE_DIM   = 32;
const unsigned int BLOCK_ROWS = 8;

/// Number of repetitions inside kernels
const unsigned int REPETITIONS = 4;

/// Tolerance for floating point comparison
const float EPSILON = 1e-3f;

/// Overload the (in)equality operator for Matrix
/// Returns true if matrices are different (dimension mismatch or value mismatch)
bool operator!=(const Matrix& m1, const Matrix& m2) {
    if (m1.dimX != m2.dimX || m1.dimY != m2.dimY) {
        return true;
    }

    unsigned int numElems = m1.dimX * m1.dimY;
    for (unsigned int i = 0; i < numElems; ++i) {
        float diff = fabsf(m1.elements[i] - m2.elements[i]);
        if (diff > EPSILON) {
            return true;
        }
    }

    return false;
}

/// CUDA error checking function
void cudaErr(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/// Compute effective bandwidth in GB/s
/// timeMs      - execution time in milliseconds
/// width, height - matrix dimensions (original matrix)
/// repetitions - how many times data was processed inside the kernel
double effectiveBandwidth(double timeMs, unsigned int width, unsigned int height, unsigned int repetitions) {
    if (timeMs <= 0.0) {
        return 0.0;
    }

    double numElements = (double)width * (double)height;
    double bytesTransferred = 2.0 * numElements * sizeof(float) * (double)repetitions; // read + write
    double timeSec = timeMs / 1000.0;

    return bytesTransferred / (timeSec * 1024.0 * 1024.0 * 1024.0);
}

/// Free matrix memory on host
void cleanupMatrix(Matrix& matrix) {
    if (matrix.elements != NULL) {
        free(matrix.elements);
        matrix.elements = NULL;
    }
}

/// Generate a matrix that is initialized with random float numbers
Matrix generateMatrix(const unsigned int dimX, const unsigned int dimY) {
    Matrix matrix;
    matrix.dimX = dimX;
    matrix.dimY = dimY;

    size_t matrixSize = (size_t)dimX * (size_t)dimY * sizeof(float);
    matrix.elements = (float*)malloc(matrixSize);
    if (matrix.elements == NULL) {
        fprintf(stderr, "Fatal: failed to allocate matrix memory.\n");
        abort();
    }

    for (unsigned int i = 0; i < dimX * dimY; ++i) {
        matrix.elements[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return matrix;
}

/// Allocate matrix on device with same dimensions as hostM
void allocateDeviceMatrix(Matrix* m, const Matrix hostM) {
    m->dimX = hostM.dimX;
    m->dimY = hostM.dimY;
    size_t size = (size_t)m->dimX * (size_t)m->dimY * sizeof(float);
    cudaErr(cudaMalloc((void**)&m->elements, size));
}

/// Copy matrix data from host to device
void copyToDeviceMatrix(Matrix deviceM, const Matrix hostM) {
    size_t size = (size_t)hostM.dimX * (size_t)hostM.dimY * sizeof(float);
    cudaErr(cudaMemcpy(deviceM.elements, hostM.elements, size, cudaMemcpyHostToDevice));
}

/// Copy matrix data from device to host
void copyFromDeviceMatrix(Matrix hostM, const Matrix deviceM) {
    size_t size = (size_t)hostM.dimX * (size_t)hostM.dimY * sizeof(float);
    cudaErr(cudaMemcpy(hostM.elements, deviceM.elements, size, cudaMemcpyDeviceToHost));
}

/// Free device matrix memory
void freeDeviceMatrix(Matrix* m) {
    if (m->elements != NULL) {
        cudaErr(cudaFree(m->elements));
        m->elements = NULL;
    }
}

/// CPU matrix transpose: out = transpose(in)
/// in:  dimY x dimX   ->   out: dimX x dimY
/// Returns EXIT_FAILURE if dimensions mismatch
int matTransposeCPU(const Matrix in, Matrix out, float* timeMsOut, double* bandwidthOut) {
    if (out.dimX != in.dimY || out.dimY != in.dimX) {
        fprintf(stderr, "Matrix dimensions do not match for transpose (CPU).\n");
        return EXIT_FAILURE;
    }

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    float tStart = 0.0f;
    float tEnd   = 0.0f;

    sdkStartTimer(&timer);
    tStart = sdkGetTimerValue(&timer);

    for (unsigned int row = 0; row < in.dimY; ++row) {
        for (unsigned int col = 0; col < in.dimX; ++col) {
            unsigned int indexIn  = row * in.dimX + col;
            unsigned int indexOut = col * out.dimX + row; // out has dimX = in.dimY
            out.elements[indexOut] = in.elements[indexIn];
        }
    }

    sdkStopTimer(&timer);
    tEnd = sdkGetTimerValue(&timer);

    float elapsedMs = tEnd - tStart;

    if (timeMsOut != NULL) {
        *timeMsOut = elapsedMs;
    }
    if (bandwidthOut != NULL) {
        *bandwidthOut = effectiveBandwidth((double)elapsedMs, in.dimX, in.dimY, 1u);
    }

    return EXIT_SUCCESS;
}

/// GPU row-to-row copy kernel (global memory)
/// Performs a row-wise copy: out = in
__global__ void rowCopyKernel(const float* input, float* output,
                              unsigned int width, unsigned int height) {
    unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    if (xIndex >= width || yIndex >= height) {
        return;
    }

    unsigned int indexBase = xIndex + width * yIndex;

    for (unsigned int r = 0; r < REPETITIONS; ++r) {
        for (unsigned int k = 0; k < TILE_DIM; k += BLOCK_ROWS) {
            unsigned int y = yIndex + k;
            if (y >= height) {
                continue;
            }
            unsigned int idx = indexBase + k * width;
            output[idx] = input[idx];
        }
    }
}

/// GPU native matrix transpose kernel (no shared memory)
/// Input:  width x height
/// Output: height x width
__global__ void transposeNaiveKernel(const float* input, float* output,
                                     unsigned int width, unsigned int height) {
    unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    if (xIndex >= width || yIndex >= height) {
        return;
    }

    unsigned int indexIn  = xIndex + width * yIndex;
    unsigned int indexOut = yIndex + height * xIndex;

    for (unsigned int r = 0; r < REPETITIONS; ++r) {
        for (unsigned int k = 0; k < TILE_DIM; k += BLOCK_ROWS) {
            unsigned int y = yIndex + k;
            if (y >= height) {
                continue;
            }
            output[indexOut + k] = input[indexIn + k * width];
        }
    }
}

/// GPU transpose kernel with shared memory and padding
/// Uses a 32x32 tile with 32x8 threads and shared memory padding
/// to avoid bank conflicts and ensure coalesced global reads/writes
__global__ void transposeSharedKernel(const float* input, float* output,
                                      unsigned int width, unsigned int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    for (unsigned int r = 0; r < REPETITIONS; ++r) {
        // Global coordinates for reading
        unsigned int xIndexIn = blockIdx.x * TILE_DIM + threadIdx.x;
        unsigned int yIndexIn = blockIdx.y * TILE_DIM + threadIdx.y;

        // Load tile from global memory to shared memory
        for (unsigned int k = 0; k < TILE_DIM; k += BLOCK_ROWS) {
            unsigned int x = xIndexIn;
            unsigned int y = yIndexIn + k;

            if (x < width && y < height) {
                unsigned int indexIn = x + width * y;
                tile[threadIdx.y + k][threadIdx.x] = input[indexIn];
            }
        }

        __syncthreads();

        // Global coordinates for writing (transposed block position)
        unsigned int xIndexOut = blockIdx.y * TILE_DIM + threadIdx.x;
        unsigned int yIndexOut = blockIdx.x * TILE_DIM + threadIdx.y;

        // Store transposed tile from shared memory to global memory
        for (unsigned int k = 0; k < TILE_DIM; k += BLOCK_ROWS) {
            unsigned int x = xIndexOut;
            unsigned int y = yIndexOut + k;

            if (x < height && y < width) {
                unsigned int indexOut = x + height * y;
                output[indexOut] = tile[threadIdx.x][threadIdx.y + k];
            }
        }

        __syncthreads();
    }
}

/// Compare two matrices for equality within EPSILON
/// Returns true if matrices match
bool matricesMatch(const Matrix& m1, const Matrix& m2) {
    return !(m1 != m2);
}

/// GPU row-to-row copy wrapper
/// out must have the same dimensions as in
int matRowCopyGPU(const Matrix in, Matrix out,
                  float* timeMsOut, double* bandwidthOut) {
    if (in.dimX != out.dimX || in.dimY != out.dimY) {
        fprintf(stderr, "Matrix dimensions do not match for row copy (GPU).\n");
        return EXIT_FAILURE;
    }

    Matrix dIn, dOut;
    allocateDeviceMatrix(&dIn, in);
    allocateDeviceMatrix(&dOut, out);
    copyToDeviceMatrix(dIn, in);

    dim3 blockSize(TILE_DIM, BLOCK_ROWS);
    dim3 gridSize(
        (in.dimX + TILE_DIM - 1) / TILE_DIM,
        (in.dimY + TILE_DIM - 1) / TILE_DIM
    );

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    float tStart = 0.0f;
    float tEnd   = 0.0f;

    sdkStartTimer(&timer);
    tStart = sdkGetTimerValue(&timer);

    rowCopyKernel<<<gridSize, blockSize>>>(dIn.elements, dOut.elements, in.dimX, in.dimY);
    cudaErr(cudaGetLastError());
    cudaErr(cudaDeviceSynchronize());

    sdkStopTimer(&timer);
    tEnd = sdkGetTimerValue(&timer);

    float elapsedMs = tEnd - tStart;

    copyFromDeviceMatrix(out, dOut);

    freeDeviceMatrix(&dIn);
    freeDeviceMatrix(&dOut);

    if (timeMsOut != NULL) {
        *timeMsOut = elapsedMs;
    }
    if (bandwidthOut != NULL) {
        *bandwidthOut = effectiveBandwidth((double)elapsedMs, in.dimX, in.dimY, REPETITIONS);
    }

    return EXIT_SUCCESS;
}

/// GPU native transpose wrapper
/// out must be the transpose of in (dimX = in.dimY, dimY = in.dimX)
int matTransposeGPUNaive(const Matrix in, Matrix out,
                         float* timeMsOut, double* bandwidthOut) {
    if (out.dimX != in.dimY || out.dimY != in.dimX) {
        fprintf(stderr, "Matrix dimensions do not match for transpose (GPU naive).\n");
        return EXIT_FAILURE;
    }

    Matrix dIn, dOut;
    allocateDeviceMatrix(&dIn, in);
    allocateDeviceMatrix(&dOut, out);
    copyToDeviceMatrix(dIn, in);

    dim3 blockSize(TILE_DIM, BLOCK_ROWS);
    dim3 gridSize(
        (in.dimX + TILE_DIM - 1) / TILE_DIM,
        (in.dimY + TILE_DIM - 1) / TILE_DIM
    );

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    float tStart = 0.0f;
    float tEnd   = 0.0f;

    sdkStartTimer(&timer);
    tStart = sdkGetTimerValue(&timer);

    transposeNaiveKernel<<<gridSize, blockSize>>>(dIn.elements, dOut.elements, in.dimX, in.dimY);
    cudaErr(cudaGetLastError());
    cudaErr(cudaDeviceSynchronize());

    sdkStopTimer(&timer);
    tEnd = sdkGetTimerValue(&timer);

    float elapsedMs = tEnd - tStart;

    copyFromDeviceMatrix(out, dOut);

    freeDeviceMatrix(&dIn);
    freeDeviceMatrix(&dOut);

    if (timeMsOut != NULL) {
        *timeMsOut = elapsedMs;
    }
    if (bandwidthOut != NULL) {
        *bandwidthOut = effectiveBandwidth((double)elapsedMs, in.dimX, in.dimY, REPETITIONS);
    }

    return EXIT_SUCCESS;
}

/// GPU shared-memory transpose wrapper
/// out must be the transpose of in (dimX = in.dimY, dimY = in.dimX)
int matTransposeGPUShared(const Matrix in, Matrix out,
                          float* timeMsOut, double* bandwidthOut) {
    if (out.dimX != in.dimY || out.dimY != in.dimX) {
        fprintf(stderr, "Matrix dimensions do not match for transpose (GPU shared).\n");
        return EXIT_FAILURE;
    }

    Matrix dIn, dOut;
    allocateDeviceMatrix(&dIn, in);
    allocateDeviceMatrix(&dOut, out);
    copyToDeviceMatrix(dIn, in);

    dim3 blockSize(TILE_DIM, BLOCK_ROWS);
    dim3 gridSize(
        (in.dimX + TILE_DIM - 1) / TILE_DIM,
        (in.dimY + TILE_DIM - 1) / TILE_DIM
    );

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    float tStart = 0.0f;
    float tEnd   = 0.0f;

    sdkStartTimer(&timer);
    tStart = sdkGetTimerValue(&timer);

    transposeSharedKernel<<<gridSize, blockSize>>>(dIn.elements, dOut.elements, in.dimX, in.dimY);
    cudaErr(cudaGetLastError());
    cudaErr(cudaDeviceSynchronize());

    sdkStopTimer(&timer);
    tEnd = sdkGetTimerValue(&timer);

    float elapsedMs = tEnd - tStart;

    copyFromDeviceMatrix(out, dOut);

    freeDeviceMatrix(&dIn);
    freeDeviceMatrix(&dOut);

    if (timeMsOut != NULL) {
        *timeMsOut = elapsedMs;
    }
    if (bandwidthOut != NULL) {
        *bandwidthOut = effectiveBandwidth((double)elapsedMs, in.dimX, in.dimY, REPETITIONS);
    }

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
    unsigned int dimY = 10000;
    unsigned int dimX = 5000;

    // Allow overriding matrix dimensions from command line (dimY dimX)
    if (argc == 3) {
        dimY = (unsigned int)atoi(argv[1]);
        dimX = (unsigned int)atoi(argv[2]);
    }

    printf("Matrix size (original): %u x %u (dimY x dimX)\n\n", dimY, dimX);

    srand(0);

    // Host matrices
    Matrix M;             // original matrix
    Matrix M_copy_gpu;    // GPU row copy output
    Matrix M_trans_cpu;   // CPU transpose result
    Matrix M_trans_naive; // GPU naive transpose result
    Matrix M_trans_shared;// GPU shared transpose result

    // Generate original matrix
    M = generateMatrix(dimX, dimY);

    // Allocate copy matrix
    M_copy_gpu.dimX = dimX;
    M_copy_gpu.dimY = dimY;
    M_copy_gpu.elements = (float*)malloc((size_t)dimX * (size_t)dimY * sizeof(float));

    // Allocate transpose matrices
    unsigned int transCols = dimY;
    unsigned int transRows = dimX;

    M_trans_cpu.dimX = transCols;
    M_trans_cpu.dimY = transRows;
    M_trans_cpu.elements = (float*)malloc((size_t)transCols * (size_t)transRows * sizeof(float));

    M_trans_naive.dimX = transCols;
    M_trans_naive.dimY = transRows;
    M_trans_naive.elements = (float*)malloc((size_t)transCols * (size_t)transRows * sizeof(float));

    M_trans_shared.dimX = transCols;
    M_trans_shared.dimY = transRows;
    M_trans_shared.elements = (float*)malloc((size_t)transCols * (size_t)transRows * sizeof(float));

    if (M_copy_gpu.elements == NULL ||
        M_trans_cpu.elements == NULL ||
        M_trans_naive.elements == NULL ||
        M_trans_shared.elements == NULL) {
        fprintf(stderr, "Fatal: failed to allocate result matrices.\n");
        return EXIT_FAILURE;
    }

    cudaErr(cudaSetDevice(0));

    // 1) GPU row-wise copy
    printf("Running GPU row-to-row copy kernel...\n");
    float  timeCopyMs     = 0.0f;
    double bandwidthCopy  = 0.0;
    if (matRowCopyGPU(M, M_copy_gpu, &timeCopyMs, &bandwidthCopy) != EXIT_SUCCESS) {
        fprintf(stderr, "GPU row copy failed.\n");
        return EXIT_FAILURE;
    }
    if (!matricesMatch(M, M_copy_gpu)) {
        printf("Row copy check: matrices differ!\n");
    } else {
        printf("Row copy check: matrices match.\n");
    }
    printf("Row copy time:           %.3f ms\n", timeCopyMs);
    printf("Row copy effective BW:   %.3f GB/s\n\n", bandwidthCopy);

    // 2) CPU transpose
    printf("Running CPU matrix transpose...\n");
    float  timeCpuMs      = 0.0f;
    double bandwidthCpu   = 0.0;
    if (matTransposeCPU(M, M_trans_cpu, &timeCpuMs, &bandwidthCpu) != EXIT_SUCCESS) {
        fprintf(stderr, "CPU matrix transpose failed.\n");
        return EXIT_FAILURE;
    }
    printf("CPU transpose time:      %.3f ms\n", timeCpuMs);
    printf("CPU effective BW:        %.3f GB/s\n\n", bandwidthCpu);

    // 3) GPU native transpose
    printf("Running GPU native transpose kernel...\n");
    float  timeNaiveMs    = 0.0f;
    double bandwidthNaive = 0.0;
    if (matTransposeGPUNaive(M, M_trans_naive, &timeNaiveMs, &bandwidthNaive) != EXIT_SUCCESS) {
        fprintf(stderr, "GPU native transpose failed.\n");
        return EXIT_FAILURE;
    }
    if (!matricesMatch(M_trans_cpu, M_trans_naive)) {
        printf("Naïve GPU transpose check: matrices differ!\n");
    } else {
        printf("Naïve GPU transpose check: matrices match.\n");
    }
    printf("Naïve GPU transpose time:    %.3f ms\n", timeNaiveMs);
    printf("Naïve GPU effective BW:      %.3f GB/s\n", bandwidthNaive);
    if (timeNaiveMs > 0.0f) {
        printf("Speedup naive GPU vs CPU:    %.2fx\n\n", timeCpuMs / timeNaiveMs);
    } else {
        printf("\n");
    }

    // 4) GPU shared-memory transpose
    printf("Running GPU shared-memory transpose kernel...\n");
    float  timeSharedMs    = 0.0f;
    double bandwidthShared = 0.0;
    if (matTransposeGPUShared(M, M_trans_shared, &timeSharedMs, &bandwidthShared) != EXIT_SUCCESS) {
        fprintf(stderr, "GPU shared transpose failed.\n");
        return EXIT_FAILURE;
    }
    if (!matricesMatch(M_trans_cpu, M_trans_shared)) {
        printf("Shared GPU transpose check: matrices differ!\n");
    } else {
        printf("Shared GPU transpose check: matrices match.\n");
    }
    printf("Shared GPU transpose time:   %.3f ms\n", timeSharedMs);
    printf("Shared GPU effective BW:     %.3f GB/s\n", bandwidthShared);
    if (timeSharedMs > 0.0f) {
        printf("Speedup shared GPU vs CPU:   %.2fx\n", timeCpuMs / timeSharedMs);
        if (timeNaiveMs > 0.0f) {
            printf("Speedup shared vs naive GPU: %.2fx\n", timeNaiveMs / timeSharedMs);
        }
    }
    printf("\n");

    // Cleanup
    cleanupMatrix(M);
    cleanupMatrix(M_copy_gpu);
    cleanupMatrix(M_trans_cpu);
    cleanupMatrix(M_trans_naive);
    cleanupMatrix(M_trans_shared);

    cudaErr(cudaDeviceReset());
    return EXIT_SUCCESS;
}
