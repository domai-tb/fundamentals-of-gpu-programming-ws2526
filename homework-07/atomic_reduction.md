# HW#7

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
[41C  󰏗 Packages : 1598 (pacman)
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
[41C    Memory : 6.66 GiB / 31.27 GiB (21%)
[41C  󱦟 OS Age  : 598 days
[41C  󱫐 Uptime  : 1 hour, 25 mins
[41C└──────────────────────────────────────────┘
[41C  ● ● ● ● ● ● ● ●
[41C
`````

## Program Output

```bash
❯ ./atomic_reductions
Running CPU reduction...
CPU average time: 7.499 ms
GPU atomic reduction average time: 18.867 ms
Atomic global GPU: WARNING: mismatch (CPU=5243528.500000, GPU=5243536.500000)
GPU cascaded reduction average time: 0.187 ms
Cascaded GPU: WARNING: mismatch (CPU=5243528.500000, GPU=5243338.500000)
```

## Analysis & Explaination

### Performance discussion

#### CPU reduction

- CPU time is about 7.499 ms.
- This is a fully sequential accumulation, so performance depends on single-core throughput and memory bandwidth.
- For this workload the CPU is relatively fast, because summation is simple and the access pattern is perfectly linear (good cache and prefetch behavior).

#### GPU atomic global reduction

- GPU atomic reduction takes about 18.867 ms, which is slower than the CPU (about 0.40x speedup, i.e. a slowdown).
- This matches the lecture discussion about atomics:

  - atomic operations are slower than normal loads/stores
  - global atomics are slower than shared-memory atomics
  - performance collapses when many threads contend for a single memory location

- In this kernel, every element triggers an atomicAdd into the same global variable:

  - this creates maximal contention
  - many warps stall and serialize behind the atomic unit
  - effective parallelism is largely lost, so the GPU cannot use its throughput advantage

#### GPU cascaded reduction

- Cascaded reduction takes about 0.187 ms, which is dramatically faster:

  - speedup vs CPU is about 7.499 / 0.187 ≈ 40.1x
  - speedup vs naive atomic GPU is about 18.867 / 0.187 ≈ 100.9x

- This matches the optimization strategies from the atomic lecture:

  - aggregation and coarsening: each thread sums many elements locally in registers, reducing the number of atomic operations
  - privatization: threads first accumulate into a block-local result (shared memory), which has much lower contention
  - final merging: only one atomicAdd to global memory per block, instead of one per element

- The result is that most work happens without atomics, and the global atomic overhead becomes negligible.

### Correctness and mismatch discussion

#### Why the output warns about mismatches

- Both GPU versions print a mismatch warning compared to the CPU sum.
- This does not necessarily mean the kernels are logically incorrect.
- Floating-point addition is not associative:

  - (a + b) + c can produce a different rounded result than a + (b + c)

- On the GPU:

  - the global atomic version effectively sums elements in a non-deterministic order depending on warp scheduling
  - the cascaded version sums in a different hierarchical order (thread local, then block, then grid)

- Because the order differs, rounding differs, so the final float sum can differ slightly even when every element is included exactly once.

#### How large are the differences

- CPU: 5243528.5
- GPU atomic: 5243536.5

  - difference is +8.0
  - relative error is about 8 / 5.24e6 ≈ 1.5e-6

- GPU cascaded: 5243338.5

  - difference is -190.0
  - relative error is about 190 / 5.24e6 ≈ 3.6e-5

- These are small relative errors for a reduction over about 10.5 million floats.
- The cascaded version can differ a bit more because it changes the reduction tree and may accumulate partial sums with different rounding than the sequential CPU loop.

#### What this implies for the checker

- A strict absolute epsilon like 1e-3 is not realistic for large float reductions.
- A better correctness check is:

  - compute the reference sum in double (or use Kahan summation)
  - compare GPU float result against that with a relative tolerance (for example 1e-5 to 1e-4 depending on input magnitude)

### Conclusion

- The timings clearly demonstrate the main lesson from the atomics lecture:

  - using atomicAdd on a single global variable is a worst-case pattern and can be slower than the CPU because of extreme contention
  - the cascaded reduction avoids this bottleneck by doing most work locally and drastically reducing the number of global atomic operations

- The mismatch warnings are expected behavior for floating-point reductions done in different orders.
- For a fair correctness check, the comparison should use a double-based CPU reference and a relative tolerance instead of expecting identical float sums.
