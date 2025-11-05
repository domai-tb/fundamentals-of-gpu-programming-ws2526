#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// Global definition of test cases
const unsigned int NUM_MAT_SIZES = 5;
const unsigned int NUM_TESTCASES_PER_MAT_SIZE = 5;
const unsigned int MATRIX_DIM_X[NUM_MAT_SIZES] = {10,100,1000,500,100};
const unsigned int MATRIX_DIM_Y[NUM_MAT_SIZES] = {10,100,1000,2000,10000};

float EXECUTION_TIME_CPU, EXECUTION_TIME_GPU;

// Matriy type definition
typedef struct {
    unsigned int dimX;
    unsigned int dimY;
    float* elements; 
} Matrix;

/// Print matrix m to stdout
void printMatrix(Matrix m) {
	for(unsigned int y = 0; y < m.dimY; y++) {
		for(unsigned int x = 0; x < m.dimX; x++) {
			printf("%5.2f ", m.elements[x + (y*m.dimY)]);
		}
		printf("\n");
	}
	printf("\n");
}

/// Generate a matrix that is initialized with random float number. 
Matrix generateMatrix(const unsigned int dimX, const unsigned int dimY) {
    Matrix matrix;

    matrix.dimX = dimX;
    matrix.dimY = dimY;

    // allocate memory
    size_t matrixSize = dimX * dimY * sizeof(float);
    matrix.elements = (float*)malloc(matrixSize);

    if (matrix.elements == NULL) {
        fprintf(stderr, "Fatal: failed to allocate matrix memory.\n");
        abort();
    } 

    // init with randome values in the range of 0 to 99
    for (unsigned int i = 0; i < dimX * dimY; i++) {
        matrix.elements[i] = static_cast<float>(rand() % 100);
    }

    return matrix;
}

// Allocate matrix on device
void allocateDeviceMatrix(Matrix* m, const Matrix hostM) {
    m->dimX = hostM.dimX;
    m->dimY = hostM.dimY;
    size_t size = m->dimX * m->dimY * sizeof(float);
    cudaMalloc((void**)&m->elements, size);
}

// Copy matrix data from host to device
void copyToDeviceMatrix(Matrix deviceM, const Matrix hostM) {
    size_t size = hostM.dimX * hostM.dimY * sizeof(float);
    cudaMemcpy(deviceM.elements, hostM.elements, size, cudaMemcpyHostToDevice);
}

// Copy matrix data from device to host
void copyFromDeviceMatrix(Matrix hostM, const Matrix deviceM) {
    size_t size = hostM.dimX * hostM.dimY * sizeof(float);
    cudaMemcpy(hostM.elements, deviceM.elements, size, cudaMemcpyDeviceToHost);
}

// Free device memory
void freeDeviceMatrix(Matrix* m) {
    cudaFree(m->elements);
    m->elements = NULL;
}

/// Add two matrixs on CPU. 
/// Takes two matrixs as arguments and will calculate m1 = m1 + m2.
/// Returns EXIT_FAILURE if an error occurs.
int addMatrixsCPU(const Matrix m1, const Matrix m2, Matrix result) {
    if (m1.dimX != m2.dimX || m1.dimY != m2.dimY) {
        fprintf(stderr, "Matrix dimensions does not match.\n");
        return EXIT_FAILURE;
    }

    // Start timer
	float tStart, tEnd;
	StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
	tStart = sdkGetTimerValue(&timer);
	sdkStartTimer(&timer);

    // Matrix Addition
    for (unsigned int y = 0; y < m1.dimY; y++) {
        for (unsigned int x = 0; x < m1.dimX; x++) {
            unsigned int idx = y * m1.dimX + x;
            result.elements[idx] = m1.elements[idx] + m2.elements[idx];
        }
    }

    // Stop timer
    sdkStopTimer(&timer);
	tEnd = sdkGetTimerValue(&timer);
	// printf("   |--> CPU Execution Time:\t%fms\n", tEnd - tStart);
    EXECUTION_TIME_CPU += (tEnd - tStart);

    return EXIT_SUCCESS;
}

/// CUDA Kernel
__global__ void addMatrixsKernel(const Matrix m1, const Matrix m2, Matrix result) {
    unsigned int x = blockIdx.x * blockDim.x * threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y * threadIdx.y;
    unsigned int idx = y * m1.dimX + x;

    result.elements[idx] = m1.elements[idx] + m2.elements[idx];
}

/// Add two matrixs on GPU. 
/// Takes two matrixs as arguments and will calculate m1 = m1 + m2.
/// Returns EXIT_FAILURE if an error occurs.
int addMatrixsGPU(const Matrix m1, const Matrix m2, Matrix result) {
    if (m1.dimX != m2.dimX || m1.dimY != m2.dimY) {
        fprintf(stderr, "Matrix dimensions does not match.\n");
        return EXIT_FAILURE;
    }

    // Prepare GPU memory
    Matrix dM1, dM2, dResult;
    allocateDeviceMatrix(&dM1, m1);
    allocateDeviceMatrix(&dM2, m2);
    allocateDeviceMatrix(&dResult, result);
    copyToDeviceMatrix(dM1, m1);
    copyToDeviceMatrix(dM2, m2);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (m1.dimX + blockSize.x - 1) / blockSize.x,
        (m1.dimY + blockSize.y - 1) / blockSize.y
    );

    // Start timer
	float tStart, tEnd;
	StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
	tStart = sdkGetTimerValue(&timer);
	sdkStartTimer(&timer);

    // Launch kernel
    addMatrixsKernel<<<gridSize, blockSize>>>(dM1, dM2, dResult);

    // Stop timer
    sdkStopTimer(&timer);
	tEnd = sdkGetTimerValue(&timer);
	// printf("   |--> GPU Execution Time:\t%fms\n", tEnd - tStart);
    EXECUTION_TIME_GPU += (tEnd - tStart);

    // Sync GPU memory
    cudaDeviceSynchronize();
    copyFromDeviceMatrix(result, dResult);

    freeDeviceMatrix(&dM1);
    freeDeviceMatrix(&dM2);
    freeDeviceMatrix(&dResult);

    return EXIT_SUCCESS;
}

int main(void) {

    // Setup GPU
    cudaSetDevice(0);

    // Iterate over test cases
    for (unsigned int i = 0; i < NUM_MAT_SIZES; i++) {
        printf("%d) Test Case of %dx%d Matrix:\n", i+1, MATRIX_DIM_X[i], MATRIX_DIM_Y[i]);

        EXECUTION_TIME_CPU = 0; // Reset 
        EXECUTION_TIME_GPU = 0; // Reset 
        for (unsigned int j = 0; j < NUM_TESTCASES_PER_MAT_SIZE; j++) {
            // Generate a randomly initilized matrix
            Matrix m1 = generateMatrix(MATRIX_DIM_X[i], MATRIX_DIM_Y[i]);
            Matrix m2 = generateMatrix(MATRIX_DIM_X[i], MATRIX_DIM_Y[i]);
            Matrix resultCPU = generateMatrix(MATRIX_DIM_X[i], MATRIX_DIM_Y[i]);
            Matrix resultGPU = generateMatrix(MATRIX_DIM_X[i], MATRIX_DIM_Y[i]);

            // Perform Addition
            addMatrixsCPU(m1, m2, resultCPU);
            addMatrixsGPU(m1, m2, resultGPU);
        }

        printf("   |--> Average CPU Execution Time:\t%fms\n", EXECUTION_TIME_CPU / NUM_TESTCASES_PER_MAT_SIZE);
        printf("   |--> Average GPU Execution Time:\t%fms\n\n", EXECUTION_TIME_GPU / NUM_TESTCASES_PER_MAT_SIZE); 
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}