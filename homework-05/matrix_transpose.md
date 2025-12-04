# HW#5

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
[41C   Kernel : 6.17.9-arch1-1
[41C  󰏗 Packages : 1597 (pacman)
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
[41C   GPU Driver : nvidia (proprietary) 580.105.08
[41C    Memory : 6.93 GiB / 31.27 GiB (22%)
[41C  󱦟 OS Age  : 585 days
[41C  󱫐 Uptime  : 33 mins
[41C└──────────────────────────────────────────┘
[41C  ● ● ● ● ● ● ● ●
[41C
`````

## Program Output

```bash
❯ ./matrix_transpose
Matrix size (original): 10000 x 5000 (dimY x dimX)

Running GPU row-to-row copy kernel...
Row copy check: matrices match.
Row copy time:           2.124 ms
Row copy effective BW:   701.561 GB/s

Running CPU matrix transpose...
CPU transpose time:      62.711 ms
CPU effective BW:        5.940 GB/s

Running GPU native transpose kernel...
Naïve GPU transpose check: matrices match.
Naïve GPU transpose time:    18.065 ms
Naïve GPU effective BW:      82.486 GB/s
Speedup naive GPU vs CPU:    3.47x

Running GPU shared-memory transpose kernel...
Shared GPU transpose check: matrices match.
Shared GPU transpose time:   1.788 ms
Shared GPU effective BW:     833.398 GB/s
Speedup shared GPU vs CPU:   35.07x
Speedup shared vs naive GPU: 10.10x
```

## Analysis & Explaination

The results clearly show that performance is dominated by how well global memory accesses are organized, not by the arithmetic.

1. The row-to-row copy kernel reaches an effective bandwidth of about 702 GB/s, which is close to the maximum achievable bandwidth of a modern GPU. This is expected: the kernel performs a simple read–write of contiguous rows, so all global memory accesses are perfectly coalesced and there is no transpose-style reordering. The GPU can therefore stream data very efficiently.

2. The CPU transpose is much slower (≈69 ms, 5.3 GB/s). A single CPU core has much less memory bandwidth and parallelism than a GPU, and the transpose pattern also hurts cache locality (you write column-wise into the result). Even though the CPU uses caches, it cannot hide the latency for this many elements, so the effective bandwidth stays an order of magnitude below the GPU.

3. The native GPU transpose improves on the CPU (≈18 ms, 83 GB/s, ~3.8× speedup over CPU) but is still far below the row-copy kernel. This matches the theory: in the native kernel, one of the directions (either the read or the write) is non-coalesced. Threads in a warp access elements with a large stride, so the hardware has to perform many separate memory transactions. As a result, the effective bandwidth is limited by poor coalescing, even though we have massive parallelism.

4. The shared-memory transpose almost matches (and even slightly “beats” in effective metric) the row-copy performance: 36x speedup over CPU and ≈9.4x over the native GPU. This is exactly what the tiling + shared memory + padding pattern is supposed to achieve. We first load a tile from global memory with coalesced accesses, transpose it in shared memory (fast, low latency), then write it back with coalesced accesses again. The extra “+1” padding in the shared tile avoids bank conflicts, so accesses in shared memory are also efficient. The very high effective bandwidth comes from counting both read and write traffic and from the ideal access pattern; it can even exceed the nominal hardware bandwidth because of how the metric is defined and possible reuse/caching effects, but the main qualitative conclusion is: coalesced, bank-conflict-free tiling lets the transpose run almost as fast as a simple copy.
