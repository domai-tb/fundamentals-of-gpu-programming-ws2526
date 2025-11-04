#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>

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

    // init with randome values in the range of 0 to 9999
    for (unsigned int i = 0; i < dimX * dimY; i++) {
        matrix.elements[i] = static_cast<float>(rand() % 10000);
    }

    return matrix;
}

int main(void) {

    // Iterate over test cases
    for (unsigned int i = 0; i < NUM_TESTS; i++) {
        Matrix m = generateMatrix(MATRIX_DIM_X[i], MATRIX_DIM_Y[i]);   
        printMatrix(m);
    }

    cudaDeviceReset();
    return 0;
}