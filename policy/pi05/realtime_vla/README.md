# realtime-vla acceleration for the click-bell Pi0.5 policy

This integration keeps the OpenPI checkpoint read-only and uses the Triton
kernels from [`dexmal/realtime-vla`](https://github.com/dexmal/realtime-vla).
It also bridges OpenPI's native `paligemma_tokenizer.model` to the Hugging Face
tokenizer-shaped API expected by realtime-vla.

The tested source checkpoint is:

```text
/home/phl/workspace/RoboSynChallenge/policy/pi05/checkpoints/
pi05_base_robosynchallenge_full/pi05_click_bell_baseline/19999
```

Clone the accelerator next to this repository, then convert the checkpoint:

```bash
git clone https://github.com/dexmal/realtime-vla.git /home/phl/workspace/realtime-vla
git -C /home/phl/workspace/realtime-vla checkout b86a942

cd /home/phl/workspace/RoboSynChallenge-realtime-vla
PYTHONPATH=. /home/phl/workspace/RoboSynChallenge/policy/pi05/.venv/bin/python \
  -m policy.pi05.realtime_vla.convert_checkpoint \
  --jax-path /home/phl/workspace/RoboSynChallenge/policy/pi05/checkpoints/pi05_base_robosynchallenge_full/pi05_click_bell_baseline/19999 \
  --output checkpoints/realtime_vla/pi05_click_bell_19999.pkl \
  --prompt "click the bell"
```

Run the Triton benchmark:

```bash
PYTHONPATH=. /home/phl/workspace/RoboSynChallenge/policy/pi05/.venv/bin/python \
  -m policy.pi05.realtime_vla.benchmark \
  --checkpoint checkpoints/realtime_vla/pi05_click_bell_19999.pkl \
  --num-views 3 --chunk-size 50 --state-dim 14 \
  --prompt "click the bell"
```

Generated `.pkl` files belong under `checkpoints/`, which is gitignored.

`accelerated_policy.RealtimeVlaPi05Policy` adds the same image resizing,
quantile state/action normalization, Pi0.5 state tokenization, and dual-arm
delta-action restoration used by the existing OpenPI deployment adapter.

The normal evaluation adapter remains on JAX by default. Enable the accelerated
backend explicitly:

```bash
PYTHON_BIN=/home/phl/workspace/RoboSynChallenge/policy/pi05/.venv/bin/python \
bash policy/pi05/eval.sh click_bell random \
  pi05_base_robosynchallenge_full pi05_click_bell_baseline 0 \
  --checkpoint_id 19999 \
  --inference_backend realtime_vla \
  --checkpoint_root /home/phl/workspace/RoboSynChallenge/policy/pi05/checkpoints \
  --converted_checkpoint /home/phl/workspace/RoboSynChallenge-realtime-vla/checkpoints/realtime_vla/pi05_click_bell_19999.pkl
```

See [RESULTS.md](RESULTS.md) for measured latency, JAX consistency and the
simulator regression. Two things in `accelerated_policy.py` exist only to make
the accelerator coexist with the simulator in one process and must stay: the
CUDA graph is recorded with `capture_error_mode="thread_local"` (the upstream
default `global` mode makes DexSim's render thread abort the process), and
`pi_model.py` keeps the upstream `Pi05Inference` untouched by subclassing it.
