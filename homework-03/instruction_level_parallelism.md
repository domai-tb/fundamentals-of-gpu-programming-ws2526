# HW#3

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
[41C    Memory : 6.13 GiB / 31.27 GiB (20%)
[41C  󱦟 OS Age  : 562 days
[41C  󱫐 Uptime  : 24 mins
[41C└──────────────────────────────────────────┘
[41C  ● ● ● ● ● ● ● ●
[41C
`````

## Command Output

```
[ ILP1 ] Threads = 1 | Execution time per operation: 2.975464e-03ms
[ ILP4 ] Threads = 1 | Execution time per operation: 7.820129e-04ms
[ ILP1 ] Threads = 2 | Execution time per operation: 8.010864e-04ms
[ ILP4 ] Threads = 2 | Execution time per operation: 3.776551e-04ms
[ ILP1 ] Threads = 4 | Execution time per operation: 4.005432e-04ms
[ ILP4 ] Threads = 4 | Execution time per operation: 1.888275e-04ms
[ ILP1 ] Threads = 8 | Execution time per operation: 2.002716e-04ms
[ ILP4 ] Threads = 8 | Execution time per operation: 9.441376e-05ms
[ ILP1 ] Threads = 16 | Execution time per operation: 1.001358e-04ms
[ ILP4 ] Threads = 16 | Execution time per operation: 4.720688e-05ms
[ ILP1 ] Threads = 32 | Execution time per operation: 5.006790e-05ms
[ ILP4 ] Threads = 32 | Execution time per operation: 2.360344e-05ms
[ ILP1 ] Threads = 64 | Execution time per operation: 2.503395e-05ms
[ ILP4 ] Threads = 64 | Execution time per operation: 1.180172e-05ms
[ ILP1 ] Threads = 128 | Execution time per operation: 1.251698e-05ms
[ ILP4 ] Threads = 128 | Execution time per operation: 5.871058e-06ms
[ ILP1 ] Threads = 256 | Execution time per operation: 6.496906e-06ms
[ ILP4 ] Threads = 256 | Execution time per operation: 5.409122e-06ms
[ ILP1 ] Threads = 512 | Execution time per operation: 5.602837e-06ms
[ ILP4 ] Threads = 512 | Execution time per operation: 5.356967e-06ms
[ ILP1 ] Threads = 1024 | Execution time per operation: 5.453825e-06ms
[ ILP4 ] Threads = 1024 | Execution time per operation: 5.334616e-06ms
```

## Explaination

When running the CUDA kernel with different numbers of threads, the performance improves as the number of threads increases. This happens because the GPU can handle more threads in parallel, making better use of its hardware resources. In the case of the ILP1 kernel, each thread only performs one operation, while the ILP4 kernel performs four operations per thread. This difference makes the ILP4 kernel much more efficient because it utilizes the GPU’s execution units better.

As we increase the number of threads, we see that the execution time per operation decreases, but after a certain point, it starts to level off. This is because the GPU becomes limited by resources like memory bandwidth and execution units. When the threads become too numerous, the overhead of managing them outweighs the performance gains, leading to diminishing returns.
