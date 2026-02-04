TARGET = parallelism_streams

NVCC = nvcc

CUDA_FLAGS = -I. -I/opt/cuda/include -I/opt/cuda/samples/common -O4
CUDA_SOURCES = ${TARGET}.cu
CUDA_OBJECTS = ${TARGET}.o

all: ${TARGET}

${TARGET}: ${CUDA_OBJECTS}
	${NVCC} -o ${TARGET} ${CUDA_FLAGS} ${CUDA_OBJECTS}
	rm -f *.o *.bkp

%.o: %.cu
	${NVCC} -c ${CUDA_FLAGS} $< -o $@

clean:
	rm -f *.o *.bak *.out
	rm -f ${TARGET}
