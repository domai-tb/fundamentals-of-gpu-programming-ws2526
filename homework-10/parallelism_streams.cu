#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

/// Emulated frame dimensions (2MB per frame for float data)
const unsigned int FRAME_WIDTH  = 1024;
const unsigned int FRAME_HEIGHT = 512;
const unsigned int FRAME_ELEMS  = FRAME_WIDTH * FRAME_HEIGHT;
const size_t       FRAME_BYTES  = (size_t)FRAME_ELEMS * sizeof(float);

/// Number of frames to process
const unsigned int NUM_FRAMES = 100;

/// Streams to test (1 stream is the baseline; >1 enables overlap)
const unsigned int MAX_STREAMS = 8;

/// Kernel execution tuning (auto-calibrated in main)
const unsigned int KERNEL_ITERS_INITIAL = 256;
const unsigned int KERNEL_ITERS_MIN     = 1;
const unsigned int KERNEL_ITERS_MAX     = 16384;

/// Timing repetitions (amortize overheads)
const unsigned int NUM_REPETITIONS_CPU = 1;   // CPU reference (first frame only)
const unsigned int NUM_REPETITIONS_GPU = 20;  // full pipeline repetitions

/// Tolerance for float comparison
const float EPSILON = 1e-3f;

/// Global time accumulators (ms)
float EXECUTION_TIME_CPU_ONE_FRAME  = 0.0f;
float EXECUTION_TIME_GPU_1STREAM    = 0.0f;
float EXECUTION_TIME_GPU_NSTREAMS   = 0.0f;

/// CUDA error checking helper
void cudaErr(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/// Fill host array with deterministic pseudo-random floats in [0,1)
void fillRandom(float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = (float)(rand() & 0xFFFF) / 65535.0f;
    }
}

/// CPU frame processing (sequential analog of the GPU kernel)
void processFrameCPU(const float* in, float* out, unsigned int n, unsigned int iters) {
    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);

    sdkStartTimer(&timer);
    float tStart = sdkGetTimerValue(&timer);

    for (unsigned int r = 0; r < NUM_REPETITIONS_CPU; ++r) {
        for (unsigned int i = 0; i < n; ++i) {
            float v = in[i];

            // 4-lane ILP-like update (mirrors GPU kernel)
            float x0 = v;
            float x1 = v * 1.001f + 0.1f;
            float x2 = v * 0.999f - 0.1f;
            float x3 = v + 0.1234f;

            for (unsigned int k = 0; k < iters; ++k) {
                x0 = x0 * 1.000001f + x1 * 0.000001f + 0.00001f;
                x1 = x1 * 0.999999f + x2 * 0.000002f - 0.00002f;
                x2 = x2 * 1.000003f + x3 * 0.000003f + 0.00003f;
                x3 = x3 * 0.999997f + x0 * 0.000004f - 0.00004f;
            }

            out[i] = (x0 + x1) + (x2 + x3);
        }
    }

    sdkStopTimer(&timer);
    float tEnd = sdkGetTimerValue(&timer);

    EXECUTION_TIME_CPU_ONE_FRAME = (tEnd - tStart) / (float)NUM_REPETITIONS_CPU;
}

/// GPU kernel: emulate compute work per frame element
__global__ void processFrameKernel(const float* in, float* out, unsigned int n, unsigned int iters) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }

    float v = in[tid];

    float x0 = v;
    float x1 = v * 1.001f + 0.1f;
    float x2 = v * 0.999f - 0.1f;
    float x3 = v + 0.1234f;

    #pragma unroll 1
    for (unsigned int k = 0; k < iters; ++k) {
        x0 = x0 * 1.000001f + x1 * 0.000001f + 0.00001f;
        x1 = x1 * 0.999999f + x2 * 0.000002f - 0.00002f;
        x2 = x2 * 1.000003f + x3 * 0.000003f + 0.00003f;
        x3 = x3 * 0.999997f + x0 * 0.000004f - 0.00004f;
    }

    out[tid] = (x0 + x1) + (x2 + x3);
}

/// Compare two frames for approximate equality
bool compareFrames(const float* ref, const float* gpu, unsigned int n) {
    float maxAbs = 0.0f;
    for (unsigned int i = 0; i < n; ++i) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > maxAbs) {
            maxAbs = diff;
        }
        if (diff > EPSILON) {
            // keep scanning to get maxAbs for reporting
        }
    }

    if (maxAbs > EPSILON) {
        printf("WARNING: mismatch detected (max abs diff = %.6f, EPSILON = %.6f)\n", maxAbs, EPSILON);
        return false;
    }

    return true;
}

/// Measure per-stage times (H2D, kernel, D2H) for one frame in stream 0
void measureStageTimesOneFrame(
    const float* hInPinned,
    float* hOutPinned,
    float* dIn,
    float* dOut,
    unsigned int iters,
    float* tH2D,
    float* tKernel,
    float* tD2H
) {
    cudaEvent_t e0, e1;
    cudaErr(cudaEventCreate(&e0));
    cudaErr(cudaEventCreate(&e1));

    // H2D
    cudaErr(cudaEventRecord(e0, 0));
    cudaErr(cudaMemcpyAsync(dIn, hInPinned, FRAME_BYTES, cudaMemcpyHostToDevice, 0));
    cudaErr(cudaEventRecord(e1, 0));
    cudaErr(cudaEventSynchronize(e1));
    cudaErr(cudaEventElapsedTime(tH2D, e0, e1));

    // Kernel
    dim3 block(256);
    dim3 grid((FRAME_ELEMS + block.x - 1) / block.x);

    cudaErr(cudaEventRecord(e0, 0));
    processFrameKernel<<<grid, block, 0, 0>>>(dIn, dOut, FRAME_ELEMS, iters);
    cudaErr(cudaGetLastError());
    cudaErr(cudaEventRecord(e1, 0));
    cudaErr(cudaEventSynchronize(e1));
    cudaErr(cudaEventElapsedTime(tKernel, e0, e1));

    // D2H
    cudaErr(cudaEventRecord(e0, 0));
    cudaErr(cudaMemcpyAsync(hOutPinned, dOut, FRAME_BYTES, cudaMemcpyDeviceToHost, 0));
    cudaErr(cudaEventRecord(e1, 0));
    cudaErr(cudaEventSynchronize(e1));
    cudaErr(cudaEventElapsedTime(tD2H, e0, e1));

    cudaErr(cudaEventDestroy(e0));
    cudaErr(cudaEventDestroy(e1));
}

/// Calibrate kernel iterations so that kernel time ~ average(H2D, D2H)
unsigned int calibrateKernelIters(
    const float* hInPinned,
    float* hOutPinned,
    float* dIn,
    float* dOut
) {
    unsigned int iters = KERNEL_ITERS_INITIAL;

    for (int pass = 0; pass < 3; ++pass) {
        float tH2D = 0.0f;
        float tK   = 0.0f;
        float tD2H = 0.0f;

        measureStageTimesOneFrame(hInPinned, hOutPinned, dIn, dOut, iters, &tH2D, &tK, &tD2H);

        float target = 0.5f * (tH2D + tD2H);
        if (tK <= 1e-6f) {
            break;
        }

        float scale = target / tK;
        unsigned int newIters = (unsigned int)floorf((float)iters * scale + 0.5f);

        if (newIters < KERNEL_ITERS_MIN) newIters = KERNEL_ITERS_MIN;
        if (newIters > KERNEL_ITERS_MAX) newIters = KERNEL_ITERS_MAX;

        printf("Calibration pass %d: iters=%u | H2D=%.3f ms, Kernel=%.3f ms, D2H=%.3f ms | target=%.3f ms -> new iters=%u\n",
            pass + 1, iters, tH2D, tK, tD2H, target, newIters);

        iters = newIters;
    }

    // Final measurement print
    {
        float tH2D = 0.0f;
        float tK   = 0.0f;
        float tD2H = 0.0f;
        measureStageTimesOneFrame(hInPinned, hOutPinned, dIn, dOut, iters, &tH2D, &tK, &tD2H);

        printf("Final stage times (one frame): H2D=%.3f ms, Kernel=%.3f ms, D2H=%.3f ms (iters=%u)\n",
            tH2D, tK, tD2H, iters);
    }

    return iters;
}

/// Run pipeline with a single (default) stream: no overlap between copy and compute
float runSingleStream(
    const float* hInPinned,
    float* hOutPinned,
    float* dIn,
    float* dOut,
    unsigned int iters
) {
    cudaEvent_t start, stop;
    cudaErr(cudaEventCreate(&start));
    cudaErr(cudaEventCreate(&stop));

    dim3 block(256);
    dim3 grid((FRAME_ELEMS + block.x - 1) / block.x);

    // Warm-up (one frame)
    cudaErr(cudaMemcpyAsync(dIn, hInPinned, FRAME_BYTES, cudaMemcpyHostToDevice, 0));
    processFrameKernel<<<grid, block, 0, 0>>>(dIn, dOut, FRAME_ELEMS, iters);
    cudaErr(cudaGetLastError());
    cudaErr(cudaMemcpyAsync(hOutPinned, dOut, FRAME_BYTES, cudaMemcpyDeviceToHost, 0));
    cudaErr(cudaDeviceSynchronize());

    float totalMs = 0.0f;

    for (unsigned int r = 0; r < NUM_REPETITIONS_GPU; ++r) {
        cudaErr(cudaEventRecord(start, 0));

        for (unsigned int f = 0; f < NUM_FRAMES; ++f) {
            const float* hInFrame  = hInPinned  + (size_t)f * FRAME_ELEMS;
            float*       hOutFrame = hOutPinned + (size_t)f * FRAME_ELEMS;

            cudaErr(cudaMemcpyAsync(dIn, hInFrame, FRAME_BYTES, cudaMemcpyHostToDevice, 0));
            processFrameKernel<<<grid, block, 0, 0>>>(dIn, dOut, FRAME_ELEMS, iters);
            cudaErr(cudaGetLastError());
            cudaErr(cudaMemcpyAsync(hOutFrame, dOut, FRAME_BYTES, cudaMemcpyDeviceToHost, 0));
        }

        cudaErr(cudaEventRecord(stop, 0));
        cudaErr(cudaEventSynchronize(stop));

        float ms = 0.0f;
        cudaErr(cudaEventElapsedTime(&ms, start, stop));
        totalMs += ms;
    }

    cudaErr(cudaEventDestroy(start));
    cudaErr(cudaEventDestroy(stop));

    return totalMs / (float)NUM_REPETITIONS_GPU;
}

/// Run pipeline with N streams: overlap H2D / Kernel / D2H across frames
float runMultiStream(
    const float* hInPinned,
    float* hOutPinned,
    float** dIn,
    float** dOut,
    cudaStream_t* streams,
    unsigned int nStreams,
    unsigned int iters
) {
    cudaEvent_t start, stop;
    cudaErr(cudaEventCreate(&start));
    cudaErr(cudaEventCreate(&stop));

    dim3 block(256);
    dim3 grid((FRAME_ELEMS + block.x - 1) / block.x);

    // Warm-up (one frame per stream)
    for (unsigned int s = 0; s < nStreams; ++s) {
        cudaErr(cudaMemcpyAsync(dIn[s], hInPinned, FRAME_BYTES, cudaMemcpyHostToDevice, streams[s]));
        processFrameKernel<<<grid, block, 0, streams[s]>>>(dIn[s], dOut[s], FRAME_ELEMS, iters);
        cudaErr(cudaGetLastError());
        cudaErr(cudaMemcpyAsync(hOutPinned, dOut[s], FRAME_BYTES, cudaMemcpyDeviceToHost, streams[s]));
    }
    cudaErr(cudaDeviceSynchronize());

    float totalMs = 0.0f;

    for (unsigned int r = 0; r < NUM_REPETITIONS_GPU; ++r) {
        cudaErr(cudaEventRecord(start, 0));

        for (unsigned int f = 0; f < NUM_FRAMES; ++f) {
            unsigned int s = f % nStreams;

            const float* hInFrame  = hInPinned  + (size_t)f * FRAME_ELEMS;
            float*       hOutFrame = hOutPinned + (size_t)f * FRAME_ELEMS;

            cudaErr(cudaMemcpyAsync(dIn[s], hInFrame, FRAME_BYTES, cudaMemcpyHostToDevice, streams[s]));
            processFrameKernel<<<grid, block, 0, streams[s]>>>(dIn[s], dOut[s], FRAME_ELEMS, iters);
            cudaErr(cudaGetLastError());
            cudaErr(cudaMemcpyAsync(hOutFrame, dOut[s], FRAME_BYTES, cudaMemcpyDeviceToHost, streams[s]));
        }

        // Record stop in default stream; under legacy semantics this will not complete
        // until all previously issued work in other streams completes.
        cudaErr(cudaEventRecord(stop, 0));
        cudaErr(cudaEventSynchronize(stop));

        float ms = 0.0f;
        cudaErr(cudaEventElapsedTime(&ms, start, stop));
        totalMs += ms;
    }

    cudaErr(cudaEventDestroy(start));
    cudaErr(cudaEventDestroy(stop));

    return totalMs / (float)NUM_REPETITIONS_GPU;
}

int main(int argc, char* argv[]) {
    srand(0);

    unsigned int nStreamsToTest[MAX_STREAMS] = {1, 2, 4, 8};
    unsigned int nTests = 4;

    printf("HW#10: Streams overlap demo (NUM_FRAMES=%u, FRAME=%ux%u, %.2f MB/frame)\n",
        NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT, (double)FRAME_BYTES / (1024.0 * 1024.0));
    printf("Timing repetitions: CPU=%u (one frame), GPU=%u (full pipeline)\n\n",
        NUM_REPETITIONS_CPU, NUM_REPETITIONS_GPU);

    cudaErr(cudaSetDevice(0));

    cudaDeviceProp prop;
    cudaErr(cudaGetDeviceProperties(&prop, 0));

    int canOverlap = 0;
    cudaErr(cudaDeviceGetAttribute(&canOverlap, cudaDevAttrGpuOverlap, 0));

    printf("GPU: %s\n", prop.name);
    printf("  concurrentKernels: %d\n", prop.concurrentKernels);
    printf("  asyncEngineCount:  %d\n", prop.asyncEngineCount);
    printf("  canOverlap:        %d\n\n", canOverlap);

    // Host pinned buffers (needed for true async H2D/D2H overlap)
    float* hInPinned  = NULL;
    float* hOut1Pinned = NULL;
    float* hOutNPinned = NULL;

    size_t totalElems = (size_t)NUM_FRAMES * (size_t)FRAME_ELEMS;
    size_t totalBytes = totalElems * sizeof(float);

    cudaErr(cudaHostAlloc((void**)&hInPinned,   totalBytes, cudaHostAllocDefault));
    cudaErr(cudaHostAlloc((void**)&hOut1Pinned, totalBytes, cudaHostAllocDefault));
    cudaErr(cudaHostAlloc((void**)&hOutNPinned, totalBytes, cudaHostAllocDefault));

    // CPU reference (pageable is fine)
    float* hCPUFrameOut = (float*)malloc(FRAME_BYTES);
    if (hCPUFrameOut == NULL) {
        fprintf(stderr, "Fatal: failed to allocate CPU output buffer.\n");
        return EXIT_FAILURE;
    }

    fillRandom(hInPinned, totalElems);
    memset(hOut1Pinned, 0, totalBytes);
    memset(hOutNPinned, 0, totalBytes);

    // Single-stream device buffers
    float* dIn0  = NULL;
    float* dOut0 = NULL;
    cudaErr(cudaMalloc((void**)&dIn0,  FRAME_BYTES));
    cudaErr(cudaMalloc((void**)&dOut0, FRAME_BYTES));

    printf("Calibrating kernel iterations so that H2D ~ Kernel ~ D2H (one frame)...\n");
    unsigned int iters = calibrateKernelIters(hInPinned, hOut1Pinned, dIn0, dOut0);
    printf("\n");

    // CPU reference on first frame
    printf("Running CPU reference on first frame...\n");
    processFrameCPU(hInPinned, hCPUFrameOut, FRAME_ELEMS, iters);
    printf("CPU time (one frame): %.3f ms\n\n", EXECUTION_TIME_CPU_ONE_FRAME);

    // GPU baseline: single stream (no overlap)
    printf("Running GPU pipeline with 1 stream (no overlap between copy and compute)...\n");
    EXECUTION_TIME_GPU_1STREAM = runSingleStream(hInPinned, hOut1Pinned, dIn0, dOut0, iters);
    printf("GPU time (1 stream): %.3f ms (avg over %u runs, %u frames/run)\n",
        EXECUTION_TIME_GPU_1STREAM, NUM_REPETITIONS_GPU, NUM_FRAMES);

    // Check correctness for first frame
    printf("Correctness check (frame 0): ");
    bool ok1 = compareFrames(hCPUFrameOut, hOut1Pinned, FRAME_ELEMS);
    if (ok1) {
        printf("matches CPU reference.\n\n");
    } else {
        printf("does NOT match CPU reference.\n\n");
    }

    // Multi-stream setup (max)
    cudaStream_t streams[MAX_STREAMS];
    float* dIn[MAX_STREAMS];
    float* dOut[MAX_STREAMS];

    for (unsigned int s = 0; s < MAX_STREAMS; ++s) {
        streams[s] = 0;
        dIn[s] = NULL;
        dOut[s] = NULL;
    }

    for (unsigned int s = 0; s < MAX_STREAMS; ++s) {
        cudaErr(cudaStreamCreate(&streams[s]));
        cudaErr(cudaMalloc((void**)&dIn[s],  FRAME_BYTES));
        cudaErr(cudaMalloc((void**)&dOut[s], FRAME_BYTES));
    }

    printf("Running GPU pipeline with multiple streams (overlap enabled)...\n\n");
    printf("| Streams | GPU time [ms] | Speedup vs 1 stream |\n");
    printf("|---------|---------------|---------------------|\n");

    for (unsigned int t = 0; t < nTests; ++t) {
        unsigned int nS = nStreamsToTest[t];
        if (nS > MAX_STREAMS) {
            continue;
        }

        if (nS == 1) {
            printf("| %7u | %13.3f | %19.2fx |\n",
                nS, EXECUTION_TIME_GPU_1STREAM, 1.0f);
            continue;
        }

        EXECUTION_TIME_GPU_NSTREAMS = runMultiStream(hInPinned, hOutNPinned, dIn, dOut, streams, nS, iters);

        float speedup = EXECUTION_TIME_GPU_1STREAM / EXECUTION_TIME_GPU_NSTREAMS;

        printf("| %7u | %13.3f | %19.2fx |\n",
            nS, EXECUTION_TIME_GPU_NSTREAMS, speedup);
    }

    // Correctness check for multi-stream output (frame 0)
    printf("\nCorrectness check (multi-stream, frame 0): ");
    bool okN = compareFrames(hCPUFrameOut, hOutNPinned, FRAME_ELEMS);
    if (okN) {
        printf("matches CPU reference.\n");
    } else {
        printf("does NOT match CPU reference.\n");
    }

    // Cleanup
    free(hCPUFrameOut);

    cudaErr(cudaFree(dIn0));
    cudaErr(cudaFree(dOut0));

    for (unsigned int s = 0; s < MAX_STREAMS; ++s) {
        if (dIn[s]  != NULL) cudaErr(cudaFree(dIn[s]));
        if (dOut[s] != NULL) cudaErr(cudaFree(dOut[s]));
        if (streams[s] != 0) cudaErr(cudaStreamDestroy(streams[s]));
    }

    cudaErr(cudaFreeHost(hInPinned));
    cudaErr(cudaFreeHost(hOut1Pinned));
    cudaErr(cudaFreeHost(hOutNPinned));

    cudaErr(cudaDeviceReset());
    return EXIT_SUCCESS;
}