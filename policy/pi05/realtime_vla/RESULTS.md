# Click-bell checkpoint acceleration results

Tested on 2026-08-20 with:

- checkpoint: `pi05_click_bell_baseline/19999`
- accelerator: `dexmal/realtime-vla` commit `b86a942`
- GPU: NVIDIA GeForce RTX 4090
- PyTorch: 2.7.1+cu126
- Triton: 3.3.1
- input: three 224×224 camera views, 14-dimensional state
- output: 50-step action chunk, 14 deployed action dimensions
- prompt: `click the bell`

## Latency

| Backend | Scope | Mean | P50 | P90 | Frequency |
|---|---|---:|---:|---:|---:|
| OpenPI JAX | end-to-end | 80.89 ms | 80.47 ms | 81.93 ms | 12.36 Hz |
| realtime-vla Triton | kernels and input copies | 42.34 ms | 42.26 ms | 42.64 ms | 23.62 Hz |
| realtime-vla Triton | end-to-end | 43.26 ms | 43.17 ms | 43.76 ms | 23.12 Hz |

The end-to-end result is a 1.87× speedup and a 46.5% latency reduction.
JAX used 10 measured iterations after warmup; Triton used 30.

## Output consistency

Both backends received the same deterministic observation and 50×32 diffusion
noise. Comparison after action unnormalization and dual-arm delta restoration:

- output shape: 50×14
- mean absolute error: 0.001094
- maximum absolute error: 0.02250
- JAX action range: [-0.18917, 1.12398]
- Triton action range: [-0.18935, 1.12304]

The residual difference is expected from the accelerator's BF16 weights and
fused Triton kernels. A simulator success-rate regression should still be run
before making Triton the default deployment backend.

