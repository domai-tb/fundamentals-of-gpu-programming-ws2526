# HW#6

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
[41C    Memory : 5.48 GiB / 31.27 GiB (18%)
[41C  󱦟 OS Age  : 592 days
[41C  󱫐 Uptime  : 13 mins
[41C└──────────────────────────────────────────┘
[41C  ● ● ● ● ● ● ● ●
[41C
`````

## Program Output

```bash
❯ ./image_upscale_convolution
Image processing HW#6: FullHD (1920x1080) -> 4K (3840x2160), mask 19x19

Running CPU pipeline (upscale + convolution)...
CPU average time over 3 repetitions: 2325.754 ms

Running GPU pipeline with global memory only ...
Global memory GPU: result matches CPU reference.
GPU global memory average time over 10 repetitions: 12.344 ms
Speedup vs CPU: 188.41x

Running GPU pipeline with constant memory mask...
Const mask GPU: result matches CPU reference.
GPU constant mask average time over 10 repetitions: 8.549 ms
Speedup vs CPU:   272.07x
Speedup vs global: 1.44x

Running GPU pipeline with constant mask + texture image (texture-based interpolation)...
Texture GPU: WARNING: GPU result differs from CPU reference!
GPU texture pipeline average time over 20 repetitions: 12.414 ms
Speedup vs CPU:      187.36x
Speedup vs global:   0.99x
Speedup vs const:    0.69x

Estimated effective bandwidths (read+write of 4K image), GB/s:
  CPU:           0.027 GB/s
  GPU global:    5.006 GB/s
  GPU const:     7.229 GB/s
  GPU texture:   4.978 GB/s
```

## Analysis & Explaination

Overall, the results show that the GPU gives a huge performance boost for this kind of image processing workload, and that using constant memory for the mask pays off more than using textures for this specific access pattern.

### CPU vs. GPU

The CPU pipeline (upscale + 19×19 convolution) takes about 2325.8 ms, while the global-memory GPU version needs only ~12.3 ms. That’s roughly a **188× speedup**.
This fits the general picture from the lectures: the computation is embarrassingly parallel (each output pixel can be computed independently), and there is a lot of regular, repeated work per pixel (19×19 stencil). A GPU can launch thousands of threads that all do the same operation on different pixels and also offers much higher memory bandwidth than a single CPU core. The CPU, even with caches, just cannot feed the arithmetic units fast enough for this big 4K image plus large kernel.

### Effect of constant memory for the mask

When we move only the 19×19 mask into constant memory (image stays in global), the runtime drops from 12.34 ms to 8.55 ms, which is about a **1.44× speedup** over the pure global version.
This matches the theory:

- The mask is small and read-only.
- All threads in a warp access the same mask elements in the inner loops.
  Constant memory is optimized exactly for this broadcast pattern, so many threads can share a single cached read. Global memory, in contrast, would fetch the same coefficients again and again. However, only the mask benefits from this; the image data still comes from global memory. That’s why we get a noticeable but not gigantic speedup: the mask traffic shrinks, but the image traffic (which dominates) stays the same.

### Texture + constant memory: performance and correctness

The texture-based version with constant mask runs in about 12.4 ms, i.e. roughly like the global-memory version and clearly **slower than the constant-only kernel**. There are two interesting observations here:

#### Performance

Textures are helpful when we have 2D locality and maybe more irregular access patterns, but they are not automatically faster than coalesced global loads plus constant memory. In our case, the access pattern is still very regular (stencil around each pixel), and the texture kernel is doing more work per output pixel: it performs the convolution directly in the input domain and calls `tex2D` many times inside the inner loops. The overhead of address computations + many texture fetches seems to balance out or even outweigh the caching benefits. That explains why the texture version ends up about the same speed as global and slower than the constant-only approach.

#### Correctness difference

The texture kernel does **not** match the CPU reference (warning in the output), while the other two GPU versions do. This is not surprising if we look at what is actually computed:

- CPU/global/const pipeline: explicit 2-step process

  1. Bilinear upscale to a discrete 4K grid.
  2. Convolution on that upscaled image with zero padding.

- Texture pipeline: fused process

For each 4K output pixel, it directly samples the original FullHD image via `tex2D` with linear filtering inside the convolution loop. That means the sampling positions, interpolation behavior, and also border handling are slightly different from doing a separate discrete upscaling first.
So mathematically we have:

- CPU/global/const: `conv( upsample_bilinear(x) )`
- Texture: `∑ w(i,j) * tex2D( scaled_coord + (i,j) )`

These operations are similar but not identical; small differences add up, so the comparison with a tight epsilon fails and we see the mismatch warning. This illustrates that using texture hardware changes not only performance, but also the exact numerical result.

### Effective bandwidth comparison

The estimated effective bandwidths (based on 4K reads+writes) underline the same trend:

- CPU: ~0.027 GB/s
- GPU global: ~5.0 GB/s
- GPU const: ~7.2 GB/s (best)
- GPU texture: ~5.0 GB/s

The absolute numbers are only rough, but the relative ordering makes sense: the constant-memory kernel uses the memory system most efficiently (less redundant traffic for the mask), while the texture kernel doesn’t show a clear bandwidth advantage over global loads for this nicely structured stencil.
