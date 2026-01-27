# HW#7

Operation performed on the following hardware:

```bash
..............                                  
            ..,;:ccc,.                           domai@tbkali
          ......''';lxO.                         OS: Kali Linux 
.....''''..........,:ld;                         Kernel: x86_64 Linux 6.18.3+kali+2-amd64
           .';;;:::;,,.x,                        Uptime: 1h 14m
      ..'''.            0Xxoc:,.  ...            Packages: 5516
  ....                ,ONkc;,;cokOdc',.          Shell: zsh 5.9
 .                   OMo           ':ddo.        Resolution: 1920x1080
                    dMc               :OO;       DE: Xfce
                    0M.                 .:o.     WM: Xfwm4
                    ;Wd                          WM Theme: Kali-Purple-Dark
                     ;XO,                        GTK Theme: Kali-Purple-Dark [GTK2]
                       ,d0Odlc;,..               Icon Theme: Flat-Remix-Purple-Light
                           ..',;:cdOOd::,.       Font: Cantarell 11
                                    .:d;.':;.    Disk: 371G / 952G (41%)
                                       'd,  .'   CPU: AMD Ryzen 7 4800H with Radeon Graphics @ 16x 2.9GHz
                                         ;l   .. GPU: NVIDIA GeForce RTX 2060
                                          .o     RAM: 8919MiB / 23399MiB
                                            c   
                                            .'  
                                             .  
                                                     
```

## Program Output

```bash
└─$ ./binary_tree-based_reduction 
Running CPU reduction...
CPU average time: 12.636 ms
Using fixed configuration: blocks=120, threads=256
GPU atomic (optimized) reduction average time: 0.195 ms
Atomic optimized GPU: result matches CPU reference (5243339.000000)
GPU cascaded (__threadfence) reduction average time: 0.205 ms
Cascaded threadfence GPU: result matches CPU reference (5243339.000000)

Summary:
CPU average time: 12.636 ms
GPU atomic optimized average time: 0.195 ms
GPU cascaded threadfence average time: 0.205 ms
```

## Analysis & Explaination

### Performance

Measured averages:

- CPU: 12.636 ms
- GPU atomic (optimized): 0.195 ms
- GPU cascaded + `__threadfence` final stage: 0.205 ms

Speedups (using the CPU result as baseline):

- Atomic-optimized GPU vs CPU: 12.636 / 0.195 = 64.8×
- Threadfence GPU vs CPU: 12.636 / 0.205 = 61.6×
- Threadfence GPU vs atomic-optimized GPU: 0.205 / 0.195 ≈ 1.05× → about 5% slower Bandwidth intuition (40 MiB read)

If we treat this as mostly a streaming read + accumulation, the implied effective read bandwidth is roughly:

- Atomic-optimized GPU: ~40 MiB / 0.195 ms = ~200 GiB/s
- Threadfence GPU: ~40 MiB / 0.205 ms = ~190 GiB/s
- CPU: ~40 MiB / 12.636 ms = ~3 GiB/s

This is consistent with the general expectation that optimized GPU reductions become memory-throughput-limited and can reach a large fraction of device bandwidth when the reduction tree is efficient (coalescing, fewer syncs, unrolling/warp-level steps).

### Comparison to Atomic Kernal (HW#7)

The HW#7 naive atomic kernel is slow because all threads contend on one address, causing serialization and stalls; this is the canonical “bad atomic reduction” case. HW#8 optimized atomic and HW#7 cascaded are both in the “good” regime: one atomic per block (or similarly small atomic count). That’s why both are ~0.2 ms scale and within ~10% of each other. The CPU time being higher in HW#8 (12.636 ms vs 7.499 ms) is plausibly due to implementation differences (e.g., accumulating in double / extra casts) and/or system state (frequency scaling, background load). This does not contradict the GPU-side conclusions; it mainly changes the reported speedup factor.