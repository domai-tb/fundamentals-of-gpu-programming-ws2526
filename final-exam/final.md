# Final Exam

## Problem 1

### 1) How many warps will have control divergence?

We use square 16x16 blocks (256 threads) for maximum occupancy. A 16x16 block has 256/32 = 8 warps per block.

The mask size is 9x7, so the half-sizes are Rx = (9-1)/2 = 4 and Ry = (7-1)/2 = 3. Control divergence appears in warps that contain a mix of threads where some threads have all 9x7 tap positions inside the image, while other threads in the same warp have at least one tap position outside the image (ghost pixels do not contribute and are skipped).

With 16x16 blocks, each warp covers 16 columns x 2 rows (32 threads). We can index warps by the top-left coordinate (x0, y0) of that 16x2 region, where x0 steps by 16 and y0 steps by 2.

- x0 in {0,16,32,...,3824} (1 + 3824/16 = 240)
- y0 in {0,2,4,...,2158} (1 + 2158/2 = 1080)

So total warps: (3840\*2160)/32 = 259200.

A warp is fully interior (no divergence) if:

- x0 >= 4 and x0+15 <= 3835 -> x0 in {16,...,3808} (1 + (3808-16)/16 = 238 values)
- y0 >= 3 and y0+1 <= 2156 -> y0 in {4,6,...,2154} (1 + (2154-4)/2 = 1076 values)

Interior warps: 238\*1076 = 256088.
Divergent warps: 259200 - 256088 = 3112.

### 2) Assuming that the algorithm bottleneck is caused by global memory operations, what would be the speedup factor if the pixel data get shifted to the shared memory buffer?

Baseline (global-memory) traffic per thread:

- 9x7 = 63 neighbor pixels, each pixel has 4 floats -> 63\*4 = 252 global float loads
- output: 4 float stores

Per 16x16 block (256 threads):

- loads: 256\*252 = 64512 floats
- stores: 256\*4 = 1024 floats
- total global ops: 65536 floats

With shared memory tiling, the block loads the required input tile (including the extra surrounding pixels needed by the mask) once:

- tile size: (16+2\*Rx) x (16+2\*Ry) = (16+8) x (16+6) = 24 x 22 = 528 pixels
- global loads into shared: 528\*4 = 2112 floats
- stores unchanged: 1024 floats
- total global ops: 3136 floats

If performance is global-memory-bandwidth bound, speedup is approximately the reduction in global traffic:

- S ~= 65536 / 3136 ~= 20.9x

## Problem 2 (5 points)

### 1) Find all errors in the code

The variable `Pvalue` is used before initialization, so the accumulation starts from an undefined value and the result is undefined.

The loop buonds are wrong: k runs from 0 to Width, so the iteration with k = Width reads `M[Row*Width + Width]` and `N[Width]`, which is out of bounds.
The correct loop range is k < Width.

Width is not a multiple of blockDim.x, so in the last block there are threads with Row >= Width. Those threads access `M[Row*Width + k]` and write `P[Row]`, which is out of bounds.
A bounds check is required (if (Row < Width) ...).

There is a missing simicolon in the loop body's statement at the end of the expression `Pvalue += ...`.

### 2) How would you implement data reuse by utilizing a shared memory buffer (provide the modified code)? Assuming that the cache hierarchy does not play a significant role, by what factor will the effective global memory bandwidth increase after this optimization?

We can see that the only reusable input across different threads in the same block is the vector N: all threads in a block need the same N[k] values. The matrix M cannot be reused across threads because each thread reads a different row.

Baseline global loads per block:

- M loads: blockDim.x \* Width floats
- N loads: blockDim.x \* Width floats

So total loads: 2 \* blockDim.x \* Width floats

With shared memory reuse of N:

- M loads: blockDim.x \* Width floats (unchanged)
- N loads: Width floats (each N element loaded once per block)

So total loads: (blockDim.x + 1) \* Width floats

So the global-memory traffic reduction factor is:

- factor = (2 \* blockDim.x \* Width) / ((blockDim.x + 1) \* Width) = 2 \* blockDim.x / (blockDim.x + 1)

For blockDim.x = 256:

- factor = 2\*256 / (256 + 1) = 512/257 ~= 1.99x

#### Code

```cpp
__global__ void MatrixVecMulKernel_SharedN(float* M, float* N, float* P, int Width)
{
    int Row = blockIdx.x * blockDim.x + threadIdx.x;
    if (Row >= Width) return;

    float Pvalue = 0.0f;

    extern __shared__ float sN[]; // size = blockDim.x floats

    for (int base = 0; base < Width; base += blockDim.x) {
        int k = base + threadIdx.x;
        sN[threadIdx.x] = (k < Width) ? N[k] : 0.0f;
        __syncthreads();

        int tileN = (Width - base < blockDim.x) ? (Width - base) : blockDim.x;
        for (int j = 0; j < tileN; j++) {
            Pvalue += M[Row * Width + (base + j)] * sN[j];
        }
        __syncthreads();
    }

    P[Row] = Pvalue;
}
```

### 3) Will the “shared memory” version from 2) be compute- or memory-bound (assume that you have a RTX 3090 GPU)?

Per output element, the work is 2\*Width floating-point operations (one multiply and one add per k).

For the shared memory version, the global-memory bytes per output element are approximately:

- M: Width floats -> 4 \* Width bytes
- N: Width floats shared by blockDim.x threads -> (4\* Width)/blockDim.x bytes per output

So total bytes per output is about 4\*Width \* (1 + 1/blockDim.x)

Arithmetic intensity is therefore:

- I = (2 \* Width flops) / (4 \* Width \* (1 + 1 / blockDim.x) bytes) = 0.5 / (1 + 1 / blockDim.x)

For blockDim.x = 256:

- I = 0.5 / (1 + 1/256) = 0.5 \* 256/257 = 128/257 ~= 0.498 flops/byte

We can see that this is a very low arithmetic intensity, so the kernel remains memory-bound.

### 4) Suppose that you want to replace the “for” loop with the parallel reduction algorithm based on the binary tree concept. How should the thread-data mapping and the execution configuration be modified to get the best performance? Give the corresponding code modifications.

We change the mapping so that one whole thread block computes one output element `P[row]`.

Execution configuration for best performance:

- gridDim.x = Width (one block per row)
- blockDim.x is a power of 2 to match the binary-tree reduction pattern

Thread-data mapping:

- blockIdx.x selects the row: row = blockIdx.x
- each thread computes a partial sum over k = threadIdx.x, threadIdx.x + blockDim.x, threadIdx.x + 2\*blockDim.x and so on
- then the block reduces the blockDim.x partial sums in shared memory using a binary tree

#### Code

```cpp
__global__ void MatrixVecMulKernel_Reduce(float* M, float* N, float* P, int Width)
{
    int row = blockIdx.x;
    int tx  = threadIdx.x;

    float sum = 0.0f;

    for (int k = tx; k < Width; k += blockDim.x) {
        sum += M[row * Width + k] * N[k];
    }

    extern __shared__ float sPart[];
    sPart[tx] = sum;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tx < stride) {
            sPart[tx] += sPart[tx + stride];
        }
    }

    if (tx == 0) {
        P[row] = sPart[0];
    }
}
```

## Problem 3 (1 point)

With L1 cache disabled, global memory reads use 32-byte granularity (L2 cache line / memory segment size).

Each thread loads one float, so useful data per thread is 4 bytes. A warp has 32 threads, so useful data per warp is 32\*4 = 128 bytes.

The access pattern is `A[9*i]`, so consecutive threads in a warp access indices that differ by 9 floats, i.e. a stride of 9\*4 = 36 bytes between threads. Since 36 > 32, each thread falls into a different 32-byte memory segment, so a warp touches 32 different 32-byte segments.

- transferred bytes per warp = 32 segments \* 32 bytes/segment = 1024 bytes
- useful bytes per warp = 32 threads \* 4 bytes/thread = 128 bytes
- efficiency = useful/transferred = 128/1024 = 1/8

Maximal effective bandwidth = peak bandwidth \* efficiency:

- B_eff = 936 GB/s /8 = 117 GB/s

## Problem 4 (1 point)

Since all atomics target the same global memory variable, the operations are serialized, so throughput is approximately the inverse of the average atomic latency.

The expected latency is:

- T_avg = 0.35\*4 ns + 0.65\*100 ns = 1.4 ns + 65 ns = 66.4 ns

So the approximate throughput is:

- Throughput ~= 1 / 66.4 ns = 1 / (66.4 \* 10^(-9)) ~= 1.506 \* 10^(-7) ops/s ~= 15 million atomic ops/s (about 0.015 Gops/s)

## Problem 5 (3 points)

### 1) How many accesses to shared memory are done for each block?

We count shared memory reads and writes to `accumResult`.

Line 21 performs 1 shared memory write per thread, so 256 writes per block.

In the reduction step, for each stride, threads with tx < stride execute line 27 which corresponds to 2 shared memory reads (`accumResult[tx]` and `accumResult[stride+tx]`) and 1 shared memory write (`accumResult[tx]`), so 3 accesses per participating thread.

The participating thread counts per stride are: 128 + 64 + 32 + 16 + 8 + 4 + 2 + 1 = 255

So shared memory accesses inside the loop are:

- 3 \* 255 = 765

Line 30 is executed by all 256 threads, so it reads `accumResult[0]` 256 times from shared memory.

Total shared memory accesses per block:

- 256 (initial writes) + 765 (reduction) + 256 (final reads) = 1277

### 2) List the source code lines, if any, that cause shared memory bank conflicts.

No lines cause shared memory bank conflicts.

### 3) Identify an opportunity to reduce the bandwidth requirement on the global memory. How would you achieve this? How many accesses can you eliminate?

We can see that the line 30 is executed by all 256 threads, so each block performs 256 global stores to the same output element, although only 1 store is needed.

We can reduce global-memory bandwidth by restricting the store to a single thread:

- Line 30: `if (tx == 0) d_C[blockIdx.x] = accumResult[0];`

Across the whole launch (VECTOR_N = 2048 blocks), eliminated global stores:

- 2048 \* 255 = 522240 global stores

## Problem 6 (2 points)

### 1) What would happen if you removed the last `__synchthreads()` in the algorithm?

If the last `__syncthreads()` is removed, we can see a race condition between:

- threads that are still updating `temp` during the last down-sweep step, and
- threads that already start reading `temp` to write `g_odata`.

As a result, some threads may write old values from `temp` to `g_odata`, so the scan output becomes incorrect.

### 2) Assume that you have 2048 elements in each section and warp size is 32, how many warps in each block will have control divergence during the reduction tree phase iteration where stride is 16?

Each thread loads 2 elements: `temp[2*thid]` and `temp[2*thid+1]`

So for n = 2048 elements, threads per block are:

- threads = n/2 = 2048/2 = 1024

Warps per block are:

- warps = 1024/32 = 32

For the iteration where stride is 16, d is 16 too, so:

- active threads are thid in {0,1,...,15}
- inactive threads are thid in {16,17,...}

- Warp 0 has thid in {0,...,31}, so 16 threads satisfy (thid < 16) and 16 do not -> this warp diverges.
- Warps 1..31 have thid >= 32, so no thread satisfies (thid < 16) -> these warps do not diverge.

So the number of divergent warps is just one.

## Problem 7 (1 point)

We compare the per-iteration time per GPU for 1D vs 2D decomposition. The computation work per GPU is the same in both decompositions, since each GPU updates N^2/P cells, so the computation term (N^2/P)\*t_comp cancels in the comparison.

For communication we assume an interior GPU. Full-duplex means send and receive can be overlapped for a given neighbor exchange, so per neighbor exchange cost is (t_start + message_size\*t_comm).

1D decomposition:

- Each GPU holds (N/P) cells in x and N cells in y, so the interface to the left neighbor is one vertical boundary of length N, and similarly to the right neighbor
- Message size per neighbor is N values
- Number of neighbors is 2
- Tcomm_1D = 2*(t_start + N*t_comm)

2D decomposition:

- The statement k_2D^2 = N^2/P implies k_2D = N/sqrt(P), so each GPU holds k_2D x k_2D cells
- Each GPU exchanges with 4 neighbors (left, right, up, down)
- Message size per neighbor is k_2D = N/sqrt(P) values
- Tcomm_2D = 4*(t_start + (N/sqrt(P))*t_comm)

1D is better than 2D when:

- (N^2/P)\*tcomp + Tcomm_1D < (N^2/P)\*tcomp + Tcomm_2D
- 2*(t_start + N\*t_comm) < 4*(t_start + (N/sqrt(P))\*t_comm)

Solving for t_start/t_comm:

- 2\*t_start + 2\*N\*t_comm < 4\*t_start + (4N/sqrt(P))\*t_comm
- 2\*N\*t_comm - (4N/sqrt(P))\*t_comm < 2\*t_start
- N\*t_comm\*(1 - 2/sqrt(P)) < t_start
- t_start/t_comm > N\*(1 - 2/sqrt(P))

So the estimate is:

- 1D decomposition gives better performance than 2D when t_start/t_comm > N\*(1 - 2/sqrt(P)).
