# HW#9

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
[41C   Kernel : 6.18.6-arch1-1
[41C  󰏗 Packages : 1643 (pacman)
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
[41C   GPU Driver : nvidia (open source) 590.48.01
[41C    Memory : 12.45 GiB / 31.27 GiB (40%)
[41C  󱦟 OS Age  : 640 days
[41C  󱫐 Uptime  : 4 hours, 31 mins
[41C└──────────────────────────────────────────┘
[41C  ● ● ● ● ● ● ● ●
[41C
`````

## Program Output

```bash
❯ ./parallel_prefix_sum
Block scan: 512 threads -> 1024 elements per block
Timing repetitions: CPU=20, GPU=200

|   n   | CPU time [ms] | GPU time [ms] | Speedup |
|-------|---------------|---------------|---------|
| 100000 |         0.055 |         0.019 |    2.91x |
| 200000 |         0.109 |         0.030 |    3.59x |
| 300000 |         0.158 |         0.042 |    3.80x |
| 400000 |         0.195 |         0.055 |    3.53x |
| 500000 |         0.278 |         0.068 |    4.11x |
| 600000 |         0.332 |         0.080 |    4.17x |
| 700000 |         0.385 |         0.096 |    4.02x |
| 800000 |         0.448 |         0.105 |    4.25x |
| 900000 |         0.508 |         0.119 |    4.28x |
| 1000000 |         0.584 |         0.136 |    4.29x |
```

## Analysis & Explaination

### Brent–Kung work-efficient scan behavior

The GPU implementation is the standard work-efficient scan structure: up-sweep (reduce) + down-sweep, done in shared memory per block segment, and then extended to large `n` by scanning per-block totals and applying offsets.
I used [Chapter 39. Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) as additional reference.

The printed table satisfies the requirement: 10 rows covering `100000 … 1000000` with a `100000` step.

### Performance analysis of the measured times

#### Scaling with `n`

Both CPU and GPU times grow approximately linearly with `n`, which is expected for scan (overall work _O(n)_). The GPU times grow more slowly, indicating better throughput.

#### Speedup trend

Speedup increases from 2.91x (100k) to 4.29x (1M) and then stabilizes around ~4.0–4.3x for larger `n`.

Interpretation:

- For smaller `n`, fixed costs (kernel launch, extra kernels for block sums/offsets, synchronization) are a larger fraction of total time.
- As `n` grows, those fixed costs are amortized, so the speedup improves and approaches a plateau (see [Chapter 39. Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda))

The small dips (e.g., at 400k, 700k) are consistent with normal measurement noise (clock variability, OS scheduling, cache state, etc.) at sub-millisecond timings.

#### Memory-bandwidth-limited plateau

An integer scan has low arithmetic intensity: it must at least read input + write output (and does additional shared-memory traffic internally). Once `n` is large enough, performance tends to be dominated by
memory throughput rather than arithmetic, so speedup saturates.

A rough throughput estimate using only global read+write of the main array (8 bytes/element):

- At n = 1,000,000, GPU time 0.136 ms → ~58.8 GB/s effective read+write throughput.
- At n = 1,000,000, CPU time 0.584 ms → ~13.7 GB/s effective read+write throughput.

That ratio aligns with the observed ~4.29x speedup.
