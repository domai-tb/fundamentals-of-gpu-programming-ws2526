#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_functions.h>  // StopWatchInterface, sdk*Timer

/// CUDA error checking function
void cudaErr(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/// Matrix type definition
typedef struct {
    unsigned int dimX;
    unsigned int dimY;
    float* elements;
} Matrix;

/// BlockSize type definition
typedef struct {
    unsigned int x;
    unsigned int y;
} BlockSize;

/// Global execution time accumulators (ms)
float EXECUTION_TIME_CPU = 0.0f;
float EXECUTION_TIME_GPU_NAIVE = 0.0f;

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

    size_t matrixSize = dimX * dimY * sizeof(float);
    matrix.elements = (float*)malloc(matrixSize);
    if (matrix.elements == NULL) {
        fprintf(stderr, "Fatal: failed to allocate matrix memory.\n");
        abort();
    }

    for (unsigned int i = 0; i < dimX * dimY; i++) {
        matrix.elements[i] = (float)(rand() % 100);
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

/// CPU matrix multiplication: C = A * B
/// Returns EXIT_FAILURE if dimensions mismatch
int matMulCPU(const Matrix A, const Matrix B, Matrix C) {
    if (A.dimX != B.dimY || C.dimY != A.dimY || C.dimX != B.dimX) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication (CPU).\n");
        return EXIT_FAILURE;
    }

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    float tStart = 0.0f;
    float tEnd = 0.0f;

    sdkStartTimer(&timer);
    tStart = sdkGetTimerValue(&timer);

    for (unsigned int row = 0; row < A.dimY; ++row) {
        for (unsigned int col = 0; col < B.dimX; ++col) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < A.dimX; ++k) {
                float a = A.elements[row * A.dimX + k];
                float b = B.elements[k * B.dimX + col];
                sum += a * b;
            }
            C.elements[row * C.dimX + col] = sum;
        }
    }

    sdkStopTimer(&timer);
    tEnd = sdkGetTimerValue(&timer);
    EXECUTION_TIME_CPU += (tEnd - tStart);

    return EXIT_SUCCESS;
}

/// Naive CUDA kernel: one thread computes one element of C
__global__ void matMulNaiveKernel(const Matrix A, const Matrix B, Matrix C) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= A.dimY || col >= B.dimX) {
        return;
    }

    if (A.dimX != B.dimY) {
        return;
    }

    float sum = 0.0f;
    for (unsigned int k = 0; k < A.dimX; ++k) {
        float a = A.elements[row * A.dimX + k];
        float b = B.elements[k * B.dimX + col];
        sum += a * b;
    }

    C.elements[row * C.dimX + col] = sum;
}

/// Tiled shared-memory CUDA kernel
/// TILE_WIDTH is provided at runtime as tileWidth
__global__ void matMulTiledKernel(const Matrix A, const Matrix B, Matrix C, unsigned int tileWidth) {
    extern __shared__ float shared[];
    float* As = shared;
    float* Bs = shared + tileWidth * tileWidth;

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int row = blockIdx.y * tileWidth + ty;
    unsigned int col = blockIdx.x * tileWidth + tx;

    if (A.dimX != B.dimY) {
        return;
    }

    float value = 0.0f;
    unsigned int numTiles = (A.dimX + tileWidth - 1) / tileWidth;

    for (unsigned int t = 0; t < numTiles; ++t) {
        unsigned int aRow = row;
        unsigned int aCol = t * tileWidth + tx;
        if (aRow < A.dimY && aCol < A.dimX) {
            As[ty * tileWidth + tx] = A.elements[aRow * A.dimX + aCol];
        } else {
            As[ty * tileWidth + tx] = 0.0f;
        }

        unsigned int bRow = t * tileWidth + ty;
        unsigned int bCol = col;
        if (bRow < B.dimY && bCol < B.dimX) {
            Bs[ty * tileWidth + tx] = B.elements[bRow * B.dimX + bCol];
        } else {
            Bs[ty * tileWidth + tx] = 0.0f;
        }

        __syncthreads();

        for (unsigned int k = 0; k < tileWidth; ++k) {
            float a = As[ty * tileWidth + k];
            float b = Bs[k * tileWidth + tx];
            value += a * b;
        }

        __syncthreads();
    }

    if (row < C.dimY && col < C.dimX) {
        C.elements[row * C.dimX + col] = value;
    }
}

/// GPU naive matrix multiplication wrapper
/// Returns EXIT_FAILURE if dimensions mismatch
int matMulGPU_Naive(const Matrix A, const Matrix B, Matrix C, BlockSize bs) {
    if (A.dimX != B.dimY || C.dimY != A.dimY || C.dimX != B.dimX) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication (GPU naive).\n");
        return EXIT_FAILURE;
    }

    Matrix dA, dB, dC;
    allocateDeviceMatrix(&dA, A);
    allocateDeviceMatrix(&dB, B);
    allocateDeviceMatrix(&dC, C);
    copyToDeviceMatrix(dA, A);
    copyToDeviceMatrix(dB, B);

    dim3 blockSize(bs.x, bs.y);
    dim3 gridSize(
        (C.dimX + blockSize.x - 1) / blockSize.x,
        (C.dimY + blockSize.y - 1) / blockSize.y
    );

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    float tStart = 0.0f;
    float tEnd = 0.0f;

    sdkStartTimer(&timer);
    tStart = sdkGetTimerValue(&timer);

    matMulNaiveKernel<<<gridSize, blockSize>>>(dA, dB, dC);
    cudaErr(cudaGetLastError());
    cudaErr(cudaDeviceSynchronize());

    sdkStopTimer(&timer);
    tEnd = sdkGetTimerValue(&timer);
    EXECUTION_TIME_GPU_NAIVE += (tEnd - tStart);

    copyFromDeviceMatrix(C, dC);

    freeDeviceMatrix(&dA);
    freeDeviceMatrix(&dB);
    freeDeviceMatrix(&dC);

    return EXIT_SUCCESS;
}

/// GPU tiled matrix multiplication wrapper for a given tileWidth
/// Returns EXIT_FAILURE if dimensions mismatch
int matMulGPU_Tiled(const Matrix A, const Matrix B, Matrix C, unsigned int tileWidth, float* timeMsOut) {
    if (A.dimX != B.dimY || C.dimY != A.dimY || C.dimX != B.dimX) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication (GPU tiled).\n");
        return EXIT_FAILURE;
    }

    Matrix dA, dB, dC;
    allocateDeviceMatrix(&dA, A);
    allocateDeviceMatrix(&dB, B);
    allocateDeviceMatrix(&dC, C);
    copyToDeviceMatrix(dA, A);
    copyToDeviceMatrix(dB, B);

    dim3 blockSize(tileWidth, tileWidth);
    dim3 gridSize(
        (C.dimX + tileWidth - 1) / tileWidth,
        (C.dimY + tileWidth - 1) / tileWidth
    );

    size_t sharedMemBytes = 2 * tileWidth * tileWidth * sizeof(float);

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    float tStart = 0.0f;
    float tEnd = 0.0f;

    sdkStartTimer(&timer);
    tStart = sdkGetTimerValue(&timer);

    matMulTiledKernel<<<gridSize, blockSize, sharedMemBytes>>>(dA, dB, dC, tileWidth);
    cudaErr(cudaGetLastError());
    cudaErr(cudaDeviceSynchronize());

    sdkStopTimer(&timer);
    tEnd = sdkGetTimerValue(&timer);

    if (timeMsOut != NULL) {
        *timeMsOut = (tEnd - tStart);
    }

    copyFromDeviceMatrix(C, dC);

    freeDeviceMatrix(&dA);
    freeDeviceMatrix(&dB);
    freeDeviceMatrix(&dC);

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    unsigned int M_rows = 10000;
    unsigned int M_cols = 5000;
    unsigned int N_cols = 20000;

    if (argc == 4) {
        M_rows = (unsigned int)atoi(argv[1]);
        M_cols = (unsigned int)atoi(argv[2]);
        N_cols = (unsigned int)atoi(argv[3]);
    }

    unsigned int N_rows = M_cols;
    unsigned int P_rows = M_rows;
    unsigned int P_cols = N_cols;

    printf("Matrix sizes:\n");
    printf("  M: %u x %u\n", M_rows, M_cols);
    printf("  N: %u x %u\n", N_rows, N_cols);
    printf("  P: %u x %u\n\n", P_rows, P_cols);

    srand(0);

    Matrix M, N, P_cpu, P_gpu_naive, P_gpu_tiled;
    M = generateMatrix(M_cols, M_rows);
    N = generateMatrix(N_cols, N_rows);

    P_cpu.dimX = P_cols;
    P_cpu.dimY = P_rows;
    P_cpu.elements = (float*)malloc((size_t)P_cols * (size_t)P_rows * sizeof(float));

    P_gpu_naive.dimX = P_cols;
    P_gpu_naive.dimY = P_rows;
    P_gpu_naive.elements = (float*)malloc((size_t)P_cols * (size_t)P_rows * sizeof(float));

    P_gpu_tiled.dimX = P_cols;
    P_gpu_tiled.dimY = P_rows;
    P_gpu_tiled.elements = (float*)malloc((size_t)P_cols * (size_t)P_rows * sizeof(float));

    if (P_cpu.elements == NULL || P_gpu_naive.elements == NULL || P_gpu_tiled.elements == NULL) {
        fprintf(stderr, "Fatal: failed to allocate result matrices.\n");
        return EXIT_FAILURE;
    }

    cudaErr(cudaSetDevice(0));

    EXECUTION_TIME_CPU = 0.0f;
    EXECUTION_TIME_GPU_NAIVE = 0.0f;

    printf("Running CPU matrix multiplication...\n");
    if (matMulCPU(M, N, P_cpu) != EXIT_SUCCESS) {
        fprintf(stderr, "CPU matrix multiplication failed.\n");
        return EXIT_FAILURE;
    }
    printf("CPU time: %.3f ms\n\n", EXECUTION_TIME_CPU);

    BlockSize bs;
    bs.x = 16;
    bs.y = 16;
    printf("Running naive GPU matrix multiplication (block %ux%u)...\n", bs.x, bs.y);
    if (matMulGPU_Naive(M, N, P_gpu_naive, bs) != EXIT_SUCCESS) {
        fprintf(stderr, "GPU naive matrix multiplication failed.\n");
        return EXIT_FAILURE;
    }

    if (P_cpu != P_gpu_naive) {
        printf("Warning: CPU and naive GPU results differ!\n");
    } else {
        printf("Naive GPU result matches CPU.\n");
    }

    printf("Naive GPU time: %.3f ms\n", EXECUTION_TIME_GPU_NAIVE);
    printf("Speedup naive GPU vs CPU: %.2fx\n\n", EXECUTION_TIME_CPU / EXECUTION_TIME_GPU_NAIVE);

    const unsigned int NUM_TILE_WIDTHS = 3;
    const unsigned int TILE_WIDTHS[NUM_TILE_WIDTHS] = {8, 16, 32};

    printf("Running tiled GPU matrix multiplication with shared memory:\n");
    for (unsigned int i = 0; i < NUM_TILE_WIDTHS; ++i) {
        unsigned int tileWidth = TILE_WIDTHS[i];

        if (tileWidth * tileWidth > 1024) {
            printf("  TILE_WIDTH = %u: skipped (too many threads per block)\n", tileWidth);
            continue;
        }

        float timeMsTiled = 0.0f;

        if (matMulGPU_Tiled(M, N, P_gpu_tiled, tileWidth, &timeMsTiled) != EXIT_SUCCESS) {
            fprintf(stderr, "GPU tiled matrix multiplication failed for TILE_WIDTH = %u.\n", tileWidth);
            continue;
        }

        if (P_cpu != P_gpu_tiled) {
            printf("  TILE_WIDTH = %2u: result differs from CPU!\n", tileWidth);
        } else {
            printf("  TILE_WIDTH = %2u: result matches CPU.\n", tileWidth);
        }

        printf("    Time: %.3f ms\n", timeMsTiled);
        printf("    Speedup vs CPU:   %.2fx\n", EXECUTION_TIME_CPU / timeMsTiled);
        printf("    Speedup vs naive: %.2fx\n", EXECUTION_TIME_GPU_NAIVE / timeMsTiled);
    }

    cleanupMatrix(M);
    cleanupMatrix(N);
    cleanupMatrix(P_cpu);
    cleanupMatrix(P_gpu_naive);
    cleanupMatrix(P_gpu_tiled);

    cudaErr(cudaDeviceReset());
    return EXIT_SUCCESS;
}
