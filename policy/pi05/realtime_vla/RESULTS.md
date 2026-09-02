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
fused Triton kernels.

## Simulator regression (2026-09-01, RTX 4090)

`click_bell / clear`, checkpoint `19999`, `--seed 0` (identical episode seeds for
both backends), 3 episodes, `pi0_step=10`:

| Backend | Episodes | Per-episode outcome | Evaluator-measured inference (mean) |
|---|---:|---|---:|
| OpenPI JAX | 2/3 | success, timeout, success | 350.3 ms over 49 calls |
| realtime-vla Triton | 2/3 | success, timeout, success | 61.8 ms over 49 calls |

Outcomes match episode by episode. The evaluator timer includes observation
preprocessing and action unnormalization (`env.step` excluded); the JAX figure
is far above the 80.89 ms microbenchmark because OpenPI's policy runs its
image transforms on the CPU per call, whereas the Triton adapter does that part
in a few numpy ops.

### What had to be fixed to get the simulator run at all

Every attempt on 2026-08-20 died before the first action. Three independent
problems, all in this branch (the upstream realtime-vla clone is untouched):

1. **CUDA graph capture mode.** `Pi05Inference.__init__` records its graph with
   torch's default `global` capture mode, which turns unsafe CUDA calls from
   *any* thread in the process into errors. DexSim's hybrid renderer keeps a
   thread that calls `cudaStreamSynchronize`, so the capture made it fail and
   the simulator aborted the process (`DFGpuSemaphore.cpp:346: CUDA stream
   synchronization failed`). `accelerated_policy.py` now records the graph with
   `capture_error_mode="thread_local"` (a `CUDAGraph` subclass injected through a
   `Pi05Inference` subclass). The abort also discards Python's stdout buffer,
   which is why the earlier diagnosis blamed `gym.make()`; run with
   `PYTHONUNBUFFERED=1` when chasing crashes like this.
2. **`PI0.pytorch_device` missing.** The rewritten `pi_model.py` never stored
   the attribute while the merged upstream `deploy_policy.eval` reads it for
   timing, so every episode raised on its first step -- and `env.close()`
   (`os._exit(0)` by default) swallowed the traceback. Affected both backends.
3. **`truncated.any()`** breaks now that the env returns a Python `bool`;
   `_any_true` was ported from `main`.

On multi-GPU hosts `select_cuda_device` additionally pins JAX to the
`--gpu_id` card (`jax_cuda_visible_devices`); otherwise it preallocates 75% of
every GPU in the machine. Use `EMBODICHAIN_SIM_EXIT_PROCESS=0` if you need the
metrics file: the default `os._exit(0)` in `env.close()` runs before it is
written.

