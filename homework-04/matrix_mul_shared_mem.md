# HW#4

Operation performed on the following hardware:

`````bash
                  -`
                 .o+`
                `ooo/
               `+oooo:
              `+oooooo:
              -+oooooo+:
            `/:-:++oooo+:
           `/++++/+++++++:
          `/++++++++++++++:
         `/+++ooooooooooooo/`
        ./ooosssso++osssssso+`
       .oossssso-````/ossssss+`
      -osssssso.      :ssssssso.
     :osssssss/        osssso+++.
    /ossssssss/        +ssssooo/-
  `/ossssso+/:-        -:/+osssso+-
 `+sso+:-`                 `.-/+oso:
`++:.                           `-/+/
.`                                 `/[1G[18A[41C[36m   󰄛  コンピューター
[41C┌──────────────────────────────────────────┐
[41C  󰇺 Chassis : 1.0
[41C  󰣇 OS : Arch Linux
[41C   Kernel : 6.16.10-arch1-1
[41C  󰏗 Packages : 1333 (pacman)
[41C  󰍹 Display : 1920x1080 @ 60Hz [External]
[41C  󰍹 Display : 2560x1440 @ 60Hz [External]
[41C  >_ Terminal : electron
[41C  󱗃 WM : Hyprland
[41C└──────────────────────────────────────────┘
[41C
[41C   : domai @ dpc
[41C┌──────────────────────────────────────────┐
[41C   CPU : AMD Ryzen 7 3700X @ 4.43 GHz
[41C  󰊴 GPU : NVIDIA GeForce RTX 2060 SUPER
[41C   GPU Driver : nvidia (proprietary) 580.82.09
[41C    Memory : 7.14 GiB / 31.27 GiB (23%)
[41C  󱦟 OS Age  : 571 days
[41C  󱫐 Uptime  : 17 mins
[41C└──────────────────────────────────────────┘
[41C  ● ● ● ● ● ● ● ●
[41C
`````

## Program Output

```
❯ ./matrix_mul_shared_mem
Matrix sizes:
  M: 10000 x 5000
  N: 5000 x 20000
  P: 10000 x 20000

Running CPU matrix multiplication...
CPU time: 997763.188 ms

Running naive GPU matrix multiplication (block 16x16)...
Naive GPU result matches CPU.
Naive GPU time: 3252.466 ms
Speedup naive GPU vs CPU: 306.77x

Running tiled GPU matrix multiplication with shared memory:
  TILE_WIDTH =  8: result matches CPU.
    Time: 3897.018 ms
    Speedup vs CPU:   256.03x
    Speedup vs naive: 0.83x
  TILE_WIDTH = 16: result matches CPU.
    Time: 2754.661 ms
    Speedup vs CPU:   362.21x
    Speedup vs naive: 1.18x
  TILE_WIDTH = 32: result matches CPU.
    Time: 2685.071 ms
    Speedup vs CPU:   371.60x
    Speedup vs naive: 1.21x
```

## Explaination

### With vs. Without Shared Memory

- Naive GPU: 3252.466 ms
- Shared-memory GPU (best `TILE_WIDTH`=32): 2685.071 ms
- => Speedup : 1.21 over naive

For these very large matrices (10 000×5 000 × 5 000×20 000 ≈ 10^12 multiply–adds), the computation is extremly arithmetic-heavy. The GPU spends most time performing floating-point operations, not waiting on global memory.

Shared memory reduces global memory traffic, but only benefits performance if memory is the bottleneck.
Here, the arithmetic workload is so massive that:

- The naive kernel’s global memory reuse is already good enough
- The cost of shared memory loads + synchronizations becomes comparable to the memory savings

This yields only a modest improvement of ~21% when switching to shared memory.

---

### CPU vs. GPU

- CPU: 997 763 ms (~1000 seconds, ~16.6 minutes)
- Naive GPU: 3252 ms
- Tiled GPU: 2685 ms

- => Speedup GPU naive vs CPU: 306.77
- => Speedup GPU tiled vs CPU: 371.6

The CPU implementation performs roighly 10^12 floating-point operations sequentially. Even with vectorization and cache hierarchies, my CPU achieves only a few dozen GFLOP/s in naïve C code without blocking optimizations.

In contrast, my GOU sustained ~370 GFLOP/s in the tiled version—not near theoretical peak, but still massively higher than the CPU.

Thus, the GPU outperforms the CPU by 300–370 times.

---

### Dependence on `TILE_WIDTH` Parameters

| `TILE_WIDTH` | Time [ms]         | Speedup vs naive | Speedup vs CPU |
| ------------ | ----------------- | ---------------- | -------------- |
| 8            | 3897 ms           | 0.83 (slower)    | 256            |
| 16           | 2754 ms           | 1.18             | 362            |
| 32           | 2685 ms (fastest) | 1.21             | 372            |

Performance improves with `tileWidth` because:

- Larger tiles re-use global memory more effectively
- The ratio of computation per synchronization increases
- The number of shared-memory reads per output element decreases
- Fewer tiles → fewer synchronizations and fewer block boundary loads

But `TILE_WIDTH` cannot grow indefinitely because:

- Threads per block = `TILE_WIDTH`^2

  - Tile 8 -> 64 threads
  - Tile 16 -> 256 threads
  - Tile 32 -> 1024 threads

- Larger tiles consume more shared memory
- Too-small tiles underutilize memory bandwidth and ALU pipelines

On my GPU the optimal tile size is 32, the largest valid option and the one that produces:

- the best global memory coalescing,
- maximal thread-level parallelism (full block of 1024 threads),
- high arithmetic intensity per block.

### Dependence on Matrix Size

- Small matrices -> CPU–GPU overhead & launch cost dominate -> speedup is small
- Medium matrices -> GPU becomes clearly faster
- Very large matrices (like this assignment) -> GPU runs hundreds of times faster, because both GPU occupancy and arithmetic intensity scale extremely well

---

### Are there any bank conflicts?

The shared memory access patterns inside the tiled kernel are:

```cpp
As[ty * tileWidth + k]  // All threads in a warp read the same address (broadcast)
Bs[k * tileWidth + tx]  // Threads in a warp read consecutive floats (stride-1)
```

- Access to `As`
  All threads in the warp read the same element at a time ->
  This uses the “broadcast” shared memory mode -> conflict-free.

- Access to `Bs`
  Threads in a warp read consecutive 4-byte floats ->
  Each thread hits a different bank -> no conflicts (bank width = 4 bytes).

Block sizes 16×16 and 32×32 are standard conflict-free configurations for tiled MM.

Thus, no shared memory bank conflicts occur with this kernel or tile configurations.

The fact that `TILE_WIDTH`=32 gives the best performance is consistent with conflict-free access patterns.
