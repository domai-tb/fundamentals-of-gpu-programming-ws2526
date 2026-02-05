# HW#10

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
[41C    Memory : 6.97 GiB / 31.27 GiB (22%)
[41C  󱦟 OS Age  : 648 days
[41C  󱫐 Uptime  : 1 hour, 5 mins
[41C└──────────────────────────────────────────┘
[41C  ● ● ● ● ● ● ● ●
[41C
`````

## Program Output

```bash
❯ ./parallelism_streams
Calibrating kernel iterations so that H2D ~ Kernel ~ D2H (one frame)...
Calibration pass 1: iters=256 | H2D=0.168 ms, Kernel=0.611 ms, D2H=0.161 ms | target=0.165 ms -> new iters=69
Calibration pass 2: iters=69 | H2D=0.165 ms, Kernel=0.149 ms, D2H=0.160 ms | target=0.162 ms -> new iters=75
Calibration pass 3: iters=75 | H2D=0.165 ms, Kernel=0.161 ms, D2H=0.159 ms | target=0.162 ms -> new iters=75
Final stage times (one frame): H2D=0.165 ms, Kernel=0.161 ms, D2H=0.158 ms (iters=75)

Running CPU reference on first frame...
CPU time (one frame): 111.226 ms

Running GPU pipeline with 1 stream (no overlap between copy and compute)...
GPU time (1 stream): 47.048 ms (avg over 20 runs, 100 frames/run)
Correctness: Matches CPU reference.

Running GPU pipeline with multiple streams (overlap enabled)...

| Streams | GPU time [ms] | Speedup vs 1 stream |
|---------|---------------|---------------------|
|       1 |        47.048 |                1.00x |
|       2 |        24.570 |                1.91x |
|       4 |        21.137 |                2.23x |
|       8 |        20.252 |                2.32x |
Correctness: Matches CPU reference.
```

## Analysis & Explaination

The calibration achieved nearly equal stage times per frame (H2D = 0.165 ms, kernel = 0.161 ms, D2H = 0.158 ms), which makes
the pipeline well-suited for overlap. With 1 stream the stages execute serialy, so the runtime is close to the sum of all
three costs per frame (about 0.48 ms/frame, measured 47.0 ms for 100 frames). Using multiple streams reduces the total time
substantially (24.6 ms with 2 streams, 21.1 ms with 4, 20.3 ms with 8) because transfers for some frames can run while the
kernel executes for others, approaching the expected steady-state limit where throughput is dominated by the slowest single
stage; the diminishing returns beyond 4 streams match the expectation that overlap saturates due to pipeline fill/drain
overhead, copy-engne limits, and bandwidth contention.
