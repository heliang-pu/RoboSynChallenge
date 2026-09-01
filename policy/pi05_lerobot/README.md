# LeRobot π₀.₅ (PyTorch) — `pi05_lerobot`

The Hugging Face LeRobot port of π₀.₅, wired into the RoboSynChallenge eval
interface. It tracks upstream `main` because that is where **MEM** — the
short-horizon visual and proprioceptive observation memory from
[arXiv:2603.03596](https://arxiv.org/abs/2603.03596) — landed
(huggingface/lerobot#4076).

This is a **second, independent** π₀.₅ integration. It does not replace
[`policy/pi05`](../pi05), which is the JAX openpi implementation and keeps its
own config, RTC runtime and checkpoints.

| | `policy/pi05` | `policy/pi05_lerobot` |
|---|---|---|
| Implementation | openpi (JAX) | LeRobot `PI05Policy` (PyTorch) |
| Checkpoints | openpi format | LeRobot `pretrained_model/` |
| MEM | not available | `use_visual_memory` / `use_proprioceptive_memory` |
| Async / RTC runtime | `rtc_runtime.py`, `async_mode` | upstream training-time RTC only |
| Runs in | eval Python | separate worker process |

## Why MEM changes the eval loop

MEM feeds the model a short history of observations: `memory_frames`
observations spaced `memory_stride` **dataset frames** apart. At inference the
policy keeps that history in its own ring buffer and pushes one entry per call
to `select_action`.

That only lines up with training if the buffer is fed once per environment
step. So this adapter has two execution modes:

- **`per_step`** — mirrors `lerobot-eval`: every env step encodes a fresh
  observation and calls `select_action`. The policy replans on its own
  `n_action_steps` queue, so the extra calls are cheap queue pops, not extra
  forward passes.
- **`chunk`** — one `predict_action_chunk` per horizon, like the openpi adapter.
  Cheaper, but MEM then only sees one frame per horizon and its history is
  stretched by that factor.

`pi05_lerobot_step_mode: auto` (the default) picks `per_step` when the
checkpoint enables MEM and `chunk` when it does not. Measured on a real
checkpoint with 4 env steps per horizon over 3 horizons:

| mode | env steps | observations MEM saw |
|---|---|---|
| `per_step` | 12 | 12 |
| `chunk` | 12 | 3 |

## Environment setup

Evaluation spans two Python contexts, which may but need not be the same env:

- the EmbodiChain/RoboSynChallenge Python that runs the simulator and
  `scripts/eval_policy.py`;
- the worker Python that imports LeRobot and loads the checkpoint.

LeRobot requires Python ≥ 3.12 and transformers v5, which the simulator env
usually cannot satisfy, hence the worker subprocess.

```bash
cd RoboSynChallenge
bash policy/pi05_lerobot/setup_lerobot.sh          # clones + installs .[pi,training]
export PI05_LEROBOT_ROOT=$PWD/policy/pi05_lerobot/lerobot
export PI05_LEROBOT_PYTHON=$(which python)         # only if the worker env differs
```

`setup_lerobot.sh` pins an upstream commit that is known to contain MEM and
refuses a revision without `src/lerobot/policies/pi05/memory.py`. Track the
moving tip with `LEROBOT_REF=main bash policy/pi05_lerobot/setup_lerobot.sh`.

An existing checkout works too — just point `PI05_LEROBOT_ROOT` at it.

## Finetune

```bash
bash policy/pi05_lerobot/finetune.sh \
    click_bell \
    lerobot_dataset/cobotmagic_Sim_click_bell \
    outputs/train/pi05_mem_click_bell \
    0
```

MEM is on by default. `memory_stride` defaults to the dataset's own fps read
from `meta/info.json`, which reproduces MEM's one-second spacing —
RoboSynChallenge data is **25 fps**, not the 30 fps LeRobot's default assumes.

| env var | default | meaning |
|---|---|---|
| `PI05_USE_VISUAL_MEMORY` | `true` | historical image tokens fused inside SigLIP |
| `PI05_USE_PROPRIOCEPTIVE_MEMORY` | `true` | one continuous backbone token per historical state |
| `PI05_MEMORY_FRAMES` | `6` | observations in the history |
| `PI05_MEMORY_STRIDE` | dataset fps | spacing in dataset frames |
| `PI05_BASE_MODEL` | `lerobot/pi05_base` | `--policy.pretrained_path` |
| `PI05_DTYPE` | `bfloat16` | |
| `PI05_GRADIENT_CHECKPOINTING` | `true` | MEM pushes 6 frames per camera through SigLIP |
| `PI05_BATCH_SIZE` / `PI05_STEPS` | `32` / `30000` | |
| `PI05_EMA_DECAY` | unset | set to `0.99` to mirror openpi's EMA weights |
| `PI05_AUTO_QUANTILE` | `0` | augment the dataset with q01/q99 stats if missing |
| `PI05_NOHUP` | `0` | run in background with a log file |

Extra arguments are forwarded to `lerobot-train` verbatim.

Turn MEM off for a stock π₀.₅ baseline:

```bash
PI05_USE_VISUAL_MEMORY=0 PI05_USE_PROPRIOCEPTIVE_MEMORY=0 \
    bash policy/pi05_lerobot/finetune.sh click_bell <dataset> <output_dir> 0
```

## Evaluate

```bash
bash policy/pi05_lerobot/eval.sh \
    click_bell random \
    outputs/train/pi05_mem_click_bell/checkpoints/last \
    0
```

A path containing `pretrained_model/` is accepted as well as the directory
itself. Extra arguments become `eval_policy.py` overrides:

```bash
bash policy/pi05_lerobot/eval.sh click_bell random <ckpt> 0 \
    --max_episodes 20 --eval_video_log true --pi05_lerobot_steps 5
```

### Eval config keys

| key | default | meaning |
|---|---|---|
| `pi05_lerobot_steps` | `10` | env steps per horizon, and the `n_action_steps` the worker overrides the checkpoint with |
| `pi05_lerobot_step_mode` | `auto` | `auto` / `per_step` / `chunk` |
| `pi05_lerobot_memory_stride` | `0` | `0` keeps the checkpoint's stride; set only to correct a mismatch |
| `pi05_lerobot_tokenizer` | `""` | override a tokenizer path baked into the checkpoint |
| `pi05_lerobot_rescale_gripper` | `auto` | map 0-1 gripper actions onto the env's physical range |
| `pi05_lerobot_gripper_indices` | `[6, 13]` | |
| `lerobot_root` / `pi05_lerobot_python` | `""` | fall back to `PI05_LEROBOT_ROOT` / `PI05_LEROBOT_PYTHON` |

`pi05_lerobot_steps` defaults to 10 to match the openpi baseline in
`policy/pi05`, where 50-step open-loop execution scored 0/19 on `item_assembly`.

## Porting older checkpoints

Checkpoints written by an earlier LeRobot can fail to load against upstream
`main`. Two failures show up in practice, both fixable without retraining:

1. **Stale config fields.** `PI05Config` has dropped fields that older
   `config.json` files still carry (e.g. `optimizer_foreach`), and draccus
   rejects them. Strip whatever is no longer a field:

   ```bash
   python - <<'PY'
   import dataclasses, json
   from lerobot.policies.pi05.configuration_pi05 import PI05Config
   path = "<checkpoint>/pretrained_model/config.json"
   valid = {f.name for f in dataclasses.fields(PI05Config)} | {"type"}
   cfg = json.load(open(path))
   print("dropping:", sorted(set(cfg) - valid))
   json.dump({k: v for k, v in cfg.items() if k in valid}, open(path, "w"), indent=2)
   PY
   ```

2. **A tokenizer path from the training machine.** The saved preprocessor stores
   an absolute `tokenizer_name`. Point it somewhere real at eval time instead of
   editing the checkpoint:

   ```bash
   --pi05_lerobot_tokenizer google/paligemma-3b-pt-224
   ```

π₀.₅ also normalizes state and actions with `QUANTILES`, so the dataset stats
need `q01`/`q99`. `finetune.sh` checks and prints the fix; run it yourself with:

```bash
python -m lerobot.scripts.augment_dataset_quantile_stats --repo-id <id> --root <dataset>
```

## Files

| file | role |
|---|---|
| `deploy_policy.py` | eval adapter: `get_model` / `eval` / `reset_model` / `close_model` |
| `pi05_worker.py` | subprocess that owns LeRobot and the policy; stdio JSON RPC |
| `deploy_policy.yml` | eval defaults |
| `eval.sh` / `finetune.sh` | entry points |
| `setup_lerobot.sh` | clone + install LeRobot at a MEM-capable revision |
