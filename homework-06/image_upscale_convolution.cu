#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

/// Image dimensions: FullHD -> 4K
const unsigned int IN_WIDTH  = 1920;
const unsigned int IN_HEIGHT = 1080;
const unsigned int OUT_WIDTH  = 3840;
const unsigned int OUT_HEIGHT = 2160;

/// Convolution mask properties
const unsigned int MASK_WIDTH  = 19;
const unsigned int MASK_RADIUS = MASK_WIDTH / 2; // 9
const unsigned int MASK_SIZE   = MASK_WIDTH * MASK_WIDTH;

/// Tolerance for floating point comparison
const float EPSILON = 1e-3f;

/// Number of repetitions for timing
const unsigned int NUM_REPETITIONS_CPU        = 3;
const unsigned int NUM_REPETITIONS_GPU_GLOBAL = 10;
const unsigned int NUM_REPETITIONS_GPU_CONST  = 10;
const unsigned int NUM_REPETITIONS_GPU_TEX    = 20;

/// Global execution time accumulators
float EXECUTION_TIME_CPU        = 0.0f;
float EXECUTION_TIME_GPU_GLOBAL = 0.0f;
float EXECUTION_TIME_GPU_CONST  = 0.0f;
float EXECUTION_TIME_GPU_TEX    = 0.0f;

/// Simple image type
typedef struct {
    unsigned int width;
    unsigned int height;
    float* elements;
} Image;

/// Block size type
typedef struct {
    unsigned int x;
    unsigned int y;
} BlockSize;

/// Constant memory for mask
__constant__ float d_mask_const[MASK_SIZE];

/// CUDA error checking helper (same style as your HW#4)
void cudaErr(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/// Compare two images for approximate equality
bool operator!=(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height) {
        return true;
    }

    unsigned int numElems = a.width * a.height;
    for (unsigned int i = 0; i < numElems; ++i) {
        float diff = fabsf(a.elements[i] - b.elements[i]);
        if (diff > EPSILON) {
            return true;
        }
    }

    return false;
}

/// Free image memory on host
void cleanupImage(Image& img) {
    if (img.elements != NULL) {
        free(img.elements);
        img.elements = NULL;
    }
}

/// Generate image with random floats in [0,1)
Image generateImage(const unsigned int width, const unsigned int height) {
    Image img;
    img.width  = width;
    img.height = height;

    size_t size = (size_t)width * (size_t)height * sizeof(float);
    img.elements = (float*)malloc(size);
    if (img.elements == NULL) {
        fprintf(stderr, "Fatal: failed to allocate image memory.\n");
        abort();
    }

    for (unsigned int i = 0; i < width * height; ++i) {
        img.elements[i] = (float)(rand() & 0xFFFF) / 65535.0f;
    }

    return img;
}

/// Generate a simple normalized 19x19 blur mask (box filter)
void generateMask(float* mask, unsigned int maskWidth) {
    unsigned int size = maskWidth * maskWidth;
    float value = 1.0f / (float)size;
    for (unsigned int i = 0; i < size; ++i) {
        mask[i] = value;
    }
}

/// Allocate device image with same dimensions as host image
void allocateDeviceImage(Image* dImg, const Image hostImg) {
    dImg->width  = hostImg.width;
    dImg->height = hostImg.height;
    size_t size = (size_t)dImg->width * (size_t)dImg->height * sizeof(float);
    cudaErr(cudaMalloc((void**)&dImg->elements, size));
}

/// Copy image data from host to device
void copyToDeviceImage(Image dImg, const Image hImg) {
    size_t size = (size_t)hImg.width * (size_t)hImg.height * sizeof(float);
    cudaErr(cudaMemcpy(dImg.elements, hImg.elements, size, cudaMemcpyHostToDevice));
}

/// Copy image data from device to host
void copyFromDeviceImage(Image hImg, const Image dImg) {
    size_t size = (size_t)hImg.width * (size_t)hImg.height * sizeof(float);
    cudaErr(cudaMemcpy(hImg.elements, dImg.elements, size, cudaMemcpyDeviceToHost));
}

/// Free device image memory
void freeDeviceImage(Image* dImg) {
    if (dImg->elements != NULL) {
        cudaErr(cudaFree(dImg->elements));
        dImg->elements = NULL;
    }
}

/// CPU bilinear upscaling: in -> out
void upscaleBilinearCPU(const Image in, Image out) {
    const float scaleX = (float)(in.width  - 1) / (float)(out.width  - 1);
    const float scaleY = (float)(in.height - 1) / (float)(out.height - 1);

    for (unsigned int yOut = 0; yOut < out.height; ++yOut) {
        float yIn = scaleY * (float)yOut;
        int y0 = (int)floorf(yIn);
        int y1 = y0 + 1;
        float wy = yIn - (float)y0;
        if (y1 >= (int)in.height) y1 = in.height - 1;

        for (unsigned int xOut = 0; xOut < out.width; ++xOut) {
            float xIn = scaleX * (float)xOut;
            int x0 = (int)floorf(xIn);
            int x1 = x0 + 1;
            float wx = xIn - (float)x0;
            if (x1 >= (int)in.width) x1 = in.width - 1;

            float p00 = in.elements[y0 * in.width + x0];
            float p10 = in.elements[y0 * in.width + x1];
            float p01 = in.elements[y1 * in.width + x0];
            float p11 = in.elements[y1 * in.width + x1];

            float v0 = p00 * (1.0f - wx) + p10 * wx;
            float v1 = p01 * (1.0f - wx) + p11 * wx;
            float value = v0 * (1.0f - wy) + v1 * wy;

            out.elements[yOut * out.width + xOut] = value;
        }
    }
}

/// CPU convolution with zero-padding (ghost pixels = 0)
void convolveCPU(const Image in, Image out, const float* mask, unsigned int maskWidth) {
    int radius = (int)maskWidth / 2;

    for (unsigned int y = 0; y < out.height; ++y) {
        for (unsigned int x = 0; x < out.width; ++x) {
            float sum = 0.0f;
            for (int j = -radius; j <= radius; ++j) {
                int yy = (int)y + j;
                if (yy < 0 || yy >= (int)in.height) {
                    continue;
                }
                for (int i = -radius; i <= radius; ++i) {
                    int xx = (int)x + i;
                    if (xx < 0 || xx >= (int)in.width) {
                        continue;
                    }
                    unsigned int maskX = (unsigned int)(i + radius);
                    unsigned int maskY = (unsigned int)(j + radius);
                    float w = mask[maskY * maskWidth + maskX];
                    float p = in.elements[yy * in.width + xx];
                    sum += w * p;
                }
            }
            out.elements[y * out.width + x] = sum;
        }
    }
}

/// CPU reference pipeline (upscale + convolve)
/// Measures total time over NUM_REPETITIONS_CPU
void cpuPipeline(const Image in, const float* mask, Image tmpUpscaled, Image out) {
    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);

    sdkStartTimer(&timer);
    float tStart = sdkGetTimerValue(&timer);

    for (unsigned int r = 0; r < NUM_REPETITIONS_CPU; ++r) {
        upscaleBilinearCPU(in, tmpUpscaled);
        convolveCPU(tmpUpscaled, out, mask, MASK_WIDTH);
    }

    sdkStopTimer(&timer);
    float tEnd = sdkGetTimerValue(&timer);

    EXECUTION_TIME_CPU = (tEnd - tStart) / (float)NUM_REPETITIONS_CPU;
}

/// Global-memory bilinear upscaling kernel
__global__ void upscaleBilinearKernelGlobal(const Image in, Image out) {
    unsigned int xOut = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yOut = blockIdx.y * blockDim.y + threadIdx.y;

    if (xOut >= out.width || yOut >= out.height) {
        return;
    }

    float scaleX = (float)(in.width  - 1) / (float)(out.width  - 1);
    float scaleY = (float)(in.height - 1) / (float)(out.height - 1);

    float xIn = scaleX * (float)xOut;
    float yIn = scaleY * (float)yOut;

    int x0 = (int)floorf(xIn);
    int x1 = x0 + 1;
    int y0 = (int)floorf(yIn);
    int y1 = y0 + 1;

    if (x1 >= (int)in.width)  x1 = in.width - 1;
    if (y1 >= (int)in.height) y1 = in.height - 1;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;

    float wx = xIn - (float)x0;
    float wy = yIn - (float)y0;

    float p00 = in.elements[y0 * in.width + x0];
    float p10 = in.elements[y0 * in.width + x1];
    float p01 = in.elements[y1 * in.width + x0];
    float p11 = in.elements[y1 * in.width + x1];

    float v0 = p00 * (1.0f - wx) + p10 * wx;
    float v1 = p01 * (1.0f - wx) + p11 * wx;
    float value = v0 * (1.0f - wy) + v1 * wy;

    out.elements[yOut * out.width + xOut] = value;
}

/// Global-memory convolution kernel (mask in global memory)
__global__ void convolveKernelGlobal(const Image in, Image out, const float* mask, unsigned int maskWidth) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out.width || y >= out.height) {
        return;
    }

    int radius = (int)maskWidth / 2;
    float sum = 0.0f;

    for (int j = -radius; j <= radius; ++j) {
        int yy = (int)y + j;
        if (yy < 0 || yy >= (int)in.height) {
            continue;
        }
        for (int i = -radius; i <= radius; ++i) {
            int xx = (int)x + i;
            if (xx < 0 || xx >= (int)in.width) {
                continue;
            }
            unsigned int maskX = (unsigned int)(i + radius);
            unsigned int maskY = (unsigned int)(j + radius);
            float w = mask[maskY * maskWidth + maskX];
            float p = in.elements[yy * in.width + xx];
            sum += w * p;
        }
    }

    out.elements[y * out.width + x] = sum;
}

/// Convolution kernel with mask in constant memory
__global__ void convolveKernelConst(const Image in, Image out, unsigned int maskWidth) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out.width || y >= out.height) {
        return;
    }

    int radius = (int)maskWidth / 2;
    float sum = 0.0f;

    for (int j = -radius; j <= radius; ++j) {
        int yy = (int)y + j;
        if (yy < 0 || yy >= (int)in.height) {
            continue;
        }
        for (int i = -radius; i <= radius; ++i) {
            int xx = (int)x + i;
            if (xx < 0 || xx >= (int)in.width) {
                continue;
            }
            unsigned int maskX = (unsigned int)(i + radius);
            unsigned int maskY = (unsigned int)(j + radius);
            float w = d_mask_const[maskY * maskWidth + maskX];
            float p = in.elements[yy * in.width + xx];
            sum += w * p;
        }
    }

    out.elements[y * out.width + x] = sum;
}

/// Texture-based fused upscale+convolution kernel
/// - Input: FullHD image bound as textur
/// - Output: 4K convolved image
/// Each thread computes one out pixel in the 4K grid. For each mask tap, it computes the corresponding 
/// coordinate in the upscaled image and maps it back to the input image using scale factors, then samples via tex2D<float>.
__global__ void upscaleConvolveKernelTexture(
    cudaTextureObject_t texIn,
    Image out,
    unsigned int inWidth,
    unsigned int inHeight,
    unsigned int maskWidth
) {
    unsigned int xOut = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yOut = blockIdx.y * blockDim.y + threadIdx.y;

    if (xOut >= out.width || yOut >= out.height) {
        return;
    }

    int radius = (int)maskWidth / 2;

    // Scaling between out (4K) and input
    float scaleX = (float)(inWidth  - 1) / (float)(out.width  - 1);
    float scaleY = (float)(inHeight - 1) / (float)(out.height - 1);

    float sum = 0.0f;

    // Center of current output pixel in upscaled coordinates
    float xCenterUpscaled = (float)xOut;
    float yCenterUpscaled = (float)yOut;

    for (int j = -radius; j <= radius; ++j) {
        float yNeighborUpscaled = yCenterUpscaled + (float)j;
        float yIn = scaleY * yNeighborUpscaled;

        for (int i = -radius; i <= radius; ++i) {
            float xNeighborUpscaled = xCenterUpscaled + (float)i;
            float xIn = scaleX * xNeighborUpscaled;

            unsigned int maskX = (unsigned int)(i + radius);
            unsigned int maskY = (unsigned int)(j + radius);
            float w = d_mask_const[maskY * maskWidth + maskX];

            // Using texture coordinates
            float p = tex2D<float>(texIn, xIn, yIn);

            sum += w * p;
        }
    }

    out.elements[yOut * out.width + xOut] = sum;
}

/// GPU version with global memory only.
/// Measures average time over NUM_REPETITIONS_GPU_GLOBAL.
int gpuPipelineGlobal(const Image hIn, const float* hMask, Image hOut, BlockSize bsUpscale, BlockSize bsConv) {
    Image dIn, dUpscaled, dOut;
    allocateDeviceImage(&dIn, hIn);

    Image tmpUpscaled;
    tmpUpscaled.width  = OUT_WIDTH;
    tmpUpscaled.height = OUT_HEIGHT;
    size_t upSize = (size_t)tmpUpscaled.width * (size_t)tmpUpscaled.height * sizeof(float);
    cudaErr(cudaMalloc((void**)&dUpscaled.elements, upSize));
    dUpscaled.width  = tmpUpscaled.width;
    dUpscaled.height = tmpUpscaled.height;

    Image tmpOut;
    tmpOut.width  = OUT_WIDTH;
    tmpOut.height = OUT_HEIGHT;
    size_t outSize = (size_t)tmpOut.width * (size_t)tmpOut.height * sizeof(float);
    cudaErr(cudaMalloc((void**)&dOut.elements, outSize));
    dOut.width  = tmpOut.width;
    dOut.height = tmpOut.height;

    float* dMask = NULL;
    size_t maskSizeBytes = MASK_SIZE * sizeof(float);
    cudaErr(cudaMalloc((void**)&dMask, maskSizeBytes));
    cudaErr(cudaMemcpy(dMask, hMask, maskSizeBytes, cudaMemcpyHostToDevice));

    copyToDeviceImage(dIn, hIn);

    dim3 blockUpscale(bsUpscale.x, bsUpscale.y);
    dim3 gridUpscale(
        (OUT_WIDTH  + blockUpscale.x - 1) / blockUpscale.x,
        (OUT_HEIGHT + blockUpscale.y - 1) / blockUpscale.y
    );

    dim3 blockConv(bsConv.x, bsConv.y);
    dim3 gridConv(
        (OUT_WIDTH  + blockConv.x - 1) / blockConv.x,
        (OUT_HEIGHT + blockConv.y - 1) / blockConv.y
    );

    // Warm-up
    upscaleBilinearKernelGlobal<<<gridUpscale, blockUpscale>>>(dIn, dUpscaled);
    cudaErr(cudaGetLastError());
    cudaErr(cudaDeviceSynchronize());
    convolveKernelGlobal<<<gridConv, blockConv>>>(dUpscaled, dOut, dMask, MASK_WIDTH);
    cudaErr(cudaGetLastError());
    cudaErr(cudaDeviceSynchronize());

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    float tStart = sdkGetTimerValue(&timer);

    for (unsigned int r = 0; r < NUM_REPETITIONS_GPU_GLOBAL; ++r) {
        upscaleBilinearKernelGlobal<<<gridUpscale, blockUpscale>>>(dIn, dUpscaled);
        cudaErr(cudaGetLastError());
        cudaErr(cudaDeviceSynchronize());

        convolveKernelGlobal<<<gridConv, blockConv>>>(dUpscaled, dOut, dMask, MASK_WIDTH);
        cudaErr(cudaGetLastError());
        cudaErr(cudaDeviceSynchronize());
    }

    sdkStopTimer(&timer);
    float tEnd = sdkGetTimerValue(&timer);

    EXECUTION_TIME_GPU_GLOBAL = (tEnd - tStart) / (float)NUM_REPETITIONS_GPU_GLOBAL;

    copyFromDeviceImage(hOut, dOut);

    freeDeviceImage(&dIn);
    cudaErr(cudaFree(dUpscaled.elements));
    cudaErr(cudaFree(dOut.elements));
    cudaErr(cudaFree(dMask));

    return EXIT_SUCCESS;
}

/// GPU version with constant mask + global image. 
/// Upscaling as before (global); convolution kernel reads from constant memory. Measures average time over NUM_REPETITIONS_GPU_CONST.
int gpuPipelineConstMask(const Image hIn, const float* hMask, Image hOut, BlockSize bsUpscale, BlockSize bsConv) {
    Image dIn, dUpscaled, dOut;
    allocateDeviceImage(&dIn, hIn);

    Image tmpUpscaled;
    tmpUpscaled.width  = OUT_WIDTH;
    tmpUpscaled.height = OUT_HEIGHT;
    size_t upSize = (size_t)tmpUpscaled.width * (size_t)tmpUpscaled.height * sizeof(float);
    cudaErr(cudaMalloc((void**)&dUpscaled.elements, upSize));
    dUpscaled.width  = tmpUpscaled.width;
    dUpscaled.height = tmpUpscaled.height;

    Image tmpOut;
    tmpOut.width  = OUT_WIDTH;
    tmpOut.height = OUT_HEIGHT;
    size_t outSize = (size_t)tmpOut.width * (size_t)tmpOut.height * sizeof(float);
    cudaErr(cudaMalloc((void**)&dOut.elements, outSize));
    dOut.width  = tmpOut.width;
    dOut.height = tmpOut.height;

    // Copy input image
    copyToDeviceImage(dIn, hIn);

    // Copy mask into constant memory
    size_t maskSizeBytes = MASK_SIZE * sizeof(float);
    cudaErr(cudaMemcpyToSymbol(d_mask_const, hMask, maskSizeBytes, 0, cudaMemcpyHostToDevice));

    dim3 blockUpscale(bsUpscale.x, bsUpscale.y);
    dim3 gridUpscale(
        (OUT_WIDTH  + blockUpscale.x - 1) / blockUpscale.x,
        (OUT_HEIGHT + blockUpscale.y - 1) / blockUpscale.y
    );

    dim3 blockConv(bsConv.x, bsConv.y);
    dim3 gridConv(
        (OUT_WIDTH  + blockConv.x - 1) / blockConv.x,
        (OUT_HEIGHT + blockConv.y - 1) / blockConv.y
    );

    // Warm-up
    upscaleBilinearKernelGlobal<<<gridUpscale, blockUpscale>>>(dIn, dUpscaled);
    cudaErr(cudaGetLastError());
    cudaErr(cudaDeviceSynchronize());
    convolveKernelConst<<<gridConv, blockConv>>>(dUpscaled, dOut, MASK_WIDTH);
    cudaErr(cudaGetLastError());
    cudaErr(cudaDeviceSynchronize());

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    float tStart = sdkGetTimerValue(&timer);

    for (unsigned int r = 0; r < NUM_REPETITIONS_GPU_CONST; ++r) {
        upscaleBilinearKernelGlobal<<<gridUpscale, blockUpscale>>>(dIn, dUpscaled);
        cudaErr(cudaGetLastError());
        cudaErr(cudaDeviceSynchronize());

        convolveKernelConst<<<gridConv, blockConv>>>(dUpscaled, dOut, MASK_WIDTH);
        cudaErr(cudaGetLastError());
        cudaErr(cudaDeviceSynchronize());
    }

    sdkStopTimer(&timer);
    float tEnd = sdkGetTimerValue(&timer);

    EXECUTION_TIME_GPU_CONST = (tEnd - tStart) / (float)NUM_REPETITIONS_GPU_CONST;

    copyFromDeviceImage(hOut, dOut);

    freeDeviceImage(&dIn);
    cudaErr(cudaFree(dUpscaled.elements));
    cudaErr(cudaFree(dOut.elements));

    return EXIT_SUCCESS;
}

/// GPU version with constant mask + texture image + texture-based interpolation.
/// The FullHD input is stored in a CUDA array and bound to a texture object with linear filtering and border addressing. 
/// The kernel performs both upscaling and convolution in a single pass Measures average time over NUM_REPETITIONS_GPU_TEX.
int gpuPipelineTexture(const Image hIn, const float* hMask, Image hOut, BlockSize bs) {
    // Copy mask into constant memory
    size_t maskSizeBytes = MASK_SIZE * sizeof(float);
    cudaErr(cudaMemcpyToSymbol(d_mask_const, hMask, maskSizeBytes, 0, cudaMemcpyHostToDevice));

    // Create CUDA array and copy input image into it
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* cuArrayIn = NULL;
    cudaErr(cudaMallocArray(&cuArrayIn, &channelDesc, hIn.width, hIn.height));

    size_t inSizeBytes = (size_t)hIn.width * (size_t)hIn.height * sizeof(float);
    cudaErr(cudaMemcpyToArray(cuArrayIn, 0, 0, hIn.elements, inSizeBytes, cudaMemcpyHostToDevice));

    // Create texture object
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArrayIn;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;          // texture-based bilinear interpolation
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;                       // coordinates in element space

    cudaTextureObject_t texIn = 0;
    cudaErr(cudaCreateTextureObject(&texIn, &resDesc, &texDesc, NULL));

    // Allocate device output image (4K)
    Image dOut;
    dOut.width  = OUT_WIDTH;
    dOut.height = OUT_HEIGHT;
    size_t outSizeBytes = (size_t)dOut.width * (size_t)dOut.height * sizeof(float);
    cudaErr(cudaMalloc((void**)&dOut.elements, outSizeBytes));

    dim3 block(bs.x, bs.y);
    dim3 grid(
        (OUT_WIDTH  + block.x - 1) / block.x,
        (OUT_HEIGHT + block.y - 1) / block.y
    );

    // Warm-up
    upscaleConvolveKernelTexture<<<grid, block>>>(texIn, dOut, hIn.width, hIn.height, MASK_WIDTH);
    cudaErr(cudaGetLastError());
    cudaErr(cudaDeviceSynchronize());

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    float tStart = sdkGetTimerValue(&timer);

    for (unsigned int r = 0; r < NUM_REPETITIONS_GPU_TEX; ++r) {
        upscaleConvolveKernelTexture<<<grid, block>>>(texIn, dOut, hIn.width, hIn.height, MASK_WIDTH);
        cudaErr(cudaGetLastError());
        cudaErr(cudaDeviceSynchronize());
    }

    sdkStopTimer(&timer);
    float tEnd = sdkGetTimerValue(&timer);

    EXECUTION_TIME_GPU_TEX = (tEnd - tStart) / (float)NUM_REPETITIONS_GPU_TEX;

    // Copy back result
    copyFromDeviceImage(hOut, dOut);

    // Cleanup
    cudaErr(cudaDestroyTextureObject(texIn));
    cudaErr(cudaFreeArray(cuArrayIn));
    cudaErr(cudaFree(dOut.elements));

    return EXIT_SUCCESS;
}

/// Compare CPU and GPU results
void compareResults(const char* name, const Image ref, const Image gpu) {
    if (ref != gpu) {
        printf("%s: WARNING: GPU result differs from CPU reference!\n", name);
    } else {
        printf("%s: result matches CPU reference.\n", name);
    }
}

/// Compute effective bandwidth in GB/s
double effectiveBandwidthGBs(double timeMs, size_t numFloatsRead, size_t numFloatsWritten, unsigned int repetitions) {
    if (timeMs <= 0.0) {
        return 0.0;
    }
    double bytes = (double)(numFloatsRead + numFloatsWritten) * sizeof(float);
    double totalBytes = bytes * (double)repetitions;
    double seconds = timeMs / 1000.0;
    return (totalBytes / seconds) / (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char* argv[]) {
    srand(0);

    printf("Image processing HW#6: FullHD (1920x1080) -> 4K (3840x2160), mask %ux%u\n\n", MASK_WIDTH, MASK_WIDTH);

    // Host images
    Image hIn      = generateImage(IN_WIDTH, IN_HEIGHT);
    Image hUpscale;
    hUpscale.width  = OUT_WIDTH;
    hUpscale.height = OUT_HEIGHT;
    hUpscale.elements = (float*)malloc((size_t)OUT_WIDTH * (size_t)OUT_HEIGHT * sizeof(float));

    Image hCPUOut;
    hCPUOut.width  = OUT_WIDTH;
    hCPUOut.height = OUT_HEIGHT;
    hCPUOut.elements = (float*)malloc((size_t)OUT_WIDTH * (size_t)OUT_HEIGHT * sizeof(float));

    Image hGPUOutGlobal;
    hGPUOutGlobal.width  = OUT_WIDTH;
    hGPUOutGlobal.height = OUT_HEIGHT;
    hGPUOutGlobal.elements = (float*)malloc((size_t)OUT_WIDTH * (size_t)OUT_HEIGHT * sizeof(float));

    Image hGPUOutConst;
    hGPUOutConst.width  = OUT_WIDTH;
    hGPUOutConst.height = OUT_HEIGHT;
    hGPUOutConst.elements = (float*)malloc((size_t)OUT_WIDTH * (size_t)OUT_HEIGHT * sizeof(float));

    Image hGPUOutTex;
    hGPUOutTex.width  = OUT_WIDTH;
    hGPUOutTex.height = OUT_HEIGHT;
    hGPUOutTex.elements = (float*)malloc((size_t)OUT_WIDTH * (size_t)OUT_HEIGHT * sizeof(float));

    if (!hUpscale.elements || !hCPUOut.elements || !hGPUOutGlobal.elements ||
        !hGPUOutConst.elements || !hGPUOutTex.elements) {
        fprintf(stderr, "Fatal: failed to allocate host output buffers.\n");
        return EXIT_FAILURE;
    }

    // Host mask
    float* hMask = (float*)malloc(MASK_SIZE * sizeof(float));
    if (hMask == NULL) {
        fprintf(stderr, "Fatal: failed to allocate host mask.\n");
        return EXIT_FAILURE;
    }
    generateMask(hMask, MASK_WIDTH);

    cudaErr(cudaSetDevice(0));

    // CPU pipeline
    printf("Running CPU pipeline (upscale + convolution)...\n");
    cpuPipeline(hIn, hMask, hUpscale, hCPUOut);
    printf("CPU average time over %u repetitions: %.3f ms\n\n", NUM_REPETITIONS_CPU, EXECUTION_TIME_CPU);

    // GPU pipeline: global memory only
    BlockSize bsUpscale, bsConv;
    bsUpscale.x = 16; bsUpscale.y = 16;
    bsConv.x    = 16; bsConv.y    = 16;

    printf("Running GPU pipeline with global memory only (mask in global memory)...\n");
    if (gpuPipelineGlobal(hIn, hMask, hGPUOutGlobal, bsUpscale, bsConv) != EXIT_SUCCESS) {
        fprintf(stderr, "GPU global pipeline failed.\n");
        return EXIT_FAILURE;
    }
    compareResults("Global memory GPU", hCPUOut, hGPUOutGlobal);
    printf("GPU global memory average time over %u repetitions: %.3f ms\n", NUM_REPETITIONS_GPU_GLOBAL, EXECUTION_TIME_GPU_GLOBAL);
    printf("Speedup vs CPU: %.2fx\n\n", EXECUTION_TIME_CPU / EXECUTION_TIME_GPU_GLOBAL);

    // GPU pipeline: constant mask + global image
    printf("Running GPU pipeline with constant memory mask...\n");
    if (gpuPipelineConstMask(hIn, hMask, hGPUOutConst, bsUpscale, bsConv) != EXIT_SUCCESS) {
        fprintf(stderr, "GPU constant mask pipeline failed.\n");
        return EXIT_FAILURE;
    }
    compareResults("Const mask GPU", hCPUOut, hGPUOutConst);
    printf("GPU constant mask average time over %u repetitions: %.3f ms\n", NUM_REPETITIONS_GPU_CONST, EXECUTION_TIME_GPU_CONST);
    printf("Speedup vs CPU:   %.2fx\n", EXECUTION_TIME_CPU / EXECUTION_TIME_GPU_CONST);
    printf("Speedup vs global: %.2fx\n\n", EXECUTION_TIME_GPU_GLOBAL / EXECUTION_TIME_GPU_CONST);

    // GPU pipeline: constant mask + texture image + texture-based interpolation
    BlockSize bsTex;
    bsTex.x = 16;
    bsTex.y = 16;

    printf("Running GPU pipeline with constant mask + texture image (texture-based interpolation)...\n");
    if (gpuPipelineTexture(hIn, hMask, hGPUOutTex, bsTex) != EXIT_SUCCESS) {
        fprintf(stderr, "GPU texture pipeline failed.\n");
        return EXIT_FAILURE;
    }
    compareResults("Texture GPU", hCPUOut, hGPUOutTex);
    printf("GPU texture pipeline average time over %u repetitions: %.3f ms\n", NUM_REPETITIONS_GPU_TEX, EXECUTION_TIME_GPU_TEX);
    printf("Speedup vs CPU:      %.2fx\n", EXECUTION_TIME_CPU / EXECUTION_TIME_GPU_TEX);
    printf("Speedup vs global:   %.2fx\n", EXECUTION_TIME_GPU_GLOBAL / EXECUTION_TIME_GPU_TEX);
    printf("Speedup vs const:    %.2fx\n\n", EXECUTION_TIME_GPU_CONST / EXECUTION_TIME_GPU_TEX);

    // Rough effective bandwidth estimates (only for convolution, not upscaling)
    size_t numPixelsOut = (size_t)OUT_WIDTH * (size_t)OUT_HEIGHT;
    double bwCPU = effectiveBandwidthGBs(EXECUTION_TIME_CPU, numPixelsOut, numPixelsOut, 1);
    double bwGPUglobal = effectiveBandwidthGBs(EXECUTION_TIME_GPU_GLOBAL, numPixelsOut, numPixelsOut, 1);
    double bwGPUconst = effectiveBandwidthGBs(EXECUTION_TIME_GPU_CONST, numPixelsOut, numPixelsOut, 1);
    double bwGPUtex = effectiveBandwidthGBs(EXECUTION_TIME_GPU_TEX, numPixelsOut, numPixelsOut, 1);

    printf("Estimated effective bandwidths (read+write of 4K image), GB/s:\n");
    printf("  CPU:           %.3f GB/s\n", bwCPU);
    printf("  GPU global:    %.3f GB/s\n", bwGPUglobal);
    printf("  GPU const:     %.3f GB/s\n", bwGPUconst);
    printf("  GPU texture:   %.3f GB/s\n", bwGPUtex);

    // Cleanup
    cleanupImage(hIn);
    cleanupImage(hUpscale);
    cleanupImage(hCPUOut);
    cleanupImage(hGPUOutGlobal);
    cleanupImage(hGPUOutConst);
    cleanupImage(hGPUOutTex);
    free(hMask);

    cudaErr(cudaDeviceReset());
    return EXIT_SUCCESS;
}