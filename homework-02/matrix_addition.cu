#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// Global definition of test cases
const unsigned int NUM_TESTS = 5;
const unsigned int MATRIX_DIM_X[NUM_TESTS] = {10,100,1000,500,100};
const unsigned int MATRIX_DIM_Y[NUM_TESTS] = {10,100,1000,2000,10000};

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

/// Add two matrixs on CPU. 
/// Takes two matrixs as arguments and will calculate m1 = m1 + m2.
/// Returns EXIT_FAILURE if an error occurs.
int addMatrixsCPU(const Matrix m1, const Matrix m2) {
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
            m1.elements[idx] += m2.elements[idx];
        }
    }

    // Stop timer
    sdkStopTimer(&timer);
	tEnd = sdkGetTimerValue(&timer);
	printf("   |--> CPU Execution Time:\t%fms\n", tEnd - tStart);

    return EXIT_SUCCESS;
}

int main(void) {

    // Setup GPU
    cudaSetDevice(0);

    // Iterate over test cases
    for (unsigned int i = 0; i < NUM_TESTS; i++) {
        printf("%d) Test Case of %dx%d Matrix:\n", i+1, MATRIX_DIM_X[i], MATRIX_DIM_Y[i]);

        // Generate a randomly initilized matrix
        Matrix m1 = generateMatrix(MATRIX_DIM_X[i], MATRIX_DIM_Y[i]);
        Matrix m2 = generateMatrix(MATRIX_DIM_X[i], MATRIX_DIM_Y[i]);   

        addMatrixsCPU(m1, m2);
        printf("\n");
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}