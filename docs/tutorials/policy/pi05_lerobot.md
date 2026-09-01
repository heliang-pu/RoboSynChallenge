# PI0.5 (LeRobot / PyTorch)

The Hugging Face LeRobot port of π₀.₅, tracking the upstream revision that added
**MEM** — short-horizon visual and proprioceptive observation memory
([arXiv:2603.03596](https://arxiv.org/abs/2603.03596),
huggingface/lerobot#4076).

This is a separate integration from <a href="pi05.html">PI0.5 (openpi / JAX)</a>.
Both stay in the repo; they use different checkpoint formats and different eval
runtimes, and can be compared head to head on the same task and setting.

## Environment Setup

First install RoboSynChallenge and EmbodiChain following the main installation
guide. The recommended workspace layout is:

```bash
RoboSynChallenge_ws/
  EmbodiChain/
  RoboSynChallenge/
```

Evaluation spans two Python contexts:

- the EmbodiChain/RoboSynChallenge Python that runs the simulator and
  `scripts/eval_policy.py`;
- the worker Python that imports LeRobot and loads the π₀.₅ checkpoint.

They can be the same environment, but usually are not: LeRobot needs Python
≥ 3.12 and transformers v5, which the simulator environment often cannot
satisfy. The adapter therefore runs the policy in a subprocess.

```bash
cd RoboSynChallenge
bash policy/pi05_lerobot/setup_lerobot.sh
export PI05_LEROBOT_ROOT=$PWD/policy/pi05_lerobot/lerobot
export PI05_LEROBOT_PYTHON=/path/to/lerobot/env/bin/python   # if it differs
```

`setup_lerobot.sh` pins an upstream commit known to contain MEM, and aborts on a
revision without `src/lerobot/policies/pi05/memory.py`. To follow upstream:

```bash
LEROBOT_REF=main bash policy/pi05_lerobot/setup_lerobot.sh
```

An existing LeRobot checkout works too — point `PI05_LEROBOT_ROOT` at it instead
of cloning again.

## Generate RoboSynChallenge Data

See <a href="../collect_data.html">Collect Data Section</a> for details.

Training reads a RoboSynChallenge LeRobot dataset straight from
`--dataset.root`; nothing needs copying into `policy/pi05_lerobot`. Raw exports
that still name the state `observation.qpos` and the cameras `<cam>.color` are
renamed automatically by `finetune.sh`.

The dataset must be **v3.0**: LeRobot ≥ 0.6 does not read v2.1. This repo keeps
both versions — the collection pipeline writes v3.0 and converts down to v2.1 for
openpi — so point `finetune.sh` at the v3.0 copy, not the one `policy/pi05`
consumes. If you only have v2.1:

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id RoboSynChallenge/cobotmagic_Sim_click_bell \
    --root lerobot_dataset/cobotmagic_Sim_click_bell \
    --push-to-hub false
```

π₀.₅ normalizes state and actions with `QUANTILES`, so the dataset stats must
carry `q01`/`q99`. `finetune.sh` verifies this and prints the fix; run it
yourself with:

```bash
python -m lerobot.scripts.augment_dataset_quantile_stats \
    --repo-id RoboSynChallenge/cobotmagic_Sim_click_bell \
    --root lerobot_dataset/cobotmagic_Sim_click_bell
```

## Finetune Model

```bash
# bash finetune.sh <task_name> <dataset_root> <output_dir> [gpu_id] [extra_opts...]
bash policy/pi05_lerobot/finetune.sh \
    click_bell \
    lerobot_dataset/cobotmagic_Sim_click_bell \
    outputs/train/pi05_mem_click_bell \
    0
```

Visual MEM is enabled by default, finetuning from `lerobot/pi05_base`.
Proprioceptive MEM is opt-in (`PI05_USE_PROPRIOCEPTIVE_MEMORY=1`): visual MEM
reuses SigLIP's pretrained projections and adds no parameters, whereas the
proprioceptive path drops the discretized state out of the prompt and introduces
`proprio_history_proj`, which `lerobot/pi05_base` has no weights for.

MEM is not free. Measured on one RTX 4090 with 3 cameras at 480×640 and
`memory_frames=6 memory_stride=25`, a planning cycle goes from 148 ms to 242 ms
(≈1.6×) and the worker holds ≈1.2 GB more VRAM for the 126-frame inference ring
buffer. Only 24 of SigLIP's 27 layers see the history — past-frame tokens are
dropped before the language backbone — so the cost is confined to the vision
tower.

`memory_stride` counts **dataset frames**, and MEM was pre-trained on
observations one second apart. `finetune.sh` therefore reads the fps from the
dataset's `meta/info.json` and uses it as the stride: RoboSynChallenge data is
**25 fps**, so the upstream default of 30 would silently stretch the history by
20%.

Common knobs (full table in `policy/pi05_lerobot/README.md`):

```bash
PI05_MEMORY_FRAMES=6 PI05_MEMORY_STRIDE=25 \
PI05_BATCH_SIZE=32 PI05_STEPS=30000 \
    bash policy/pi05_lerobot/finetune.sh click_bell <dataset> <output_dir> 0

# stock π₀.₅ baseline, no MEM
PI05_USE_VISUAL_MEMORY=0 \
    bash policy/pi05_lerobot/finetune.sh click_bell <dataset> <output_dir> 0
```

Anything after the fourth argument is forwarded to `lerobot-train` unchanged.

## Eval on RoboSynChallenge

```bash
# bash eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]
bash policy/pi05_lerobot/eval.sh \
    click_bell random \
    outputs/train/pi05_mem_click_bell/checkpoints/last \
    0
```

A directory containing `pretrained_model/` is accepted as well as
`pretrained_model/` itself. Extra arguments become `eval_policy.py` overrides:

```bash
bash policy/pi05_lerobot/eval.sh click_bell random <ckpt> 0 \
    --max_episodes 20 --eval_video_log true
```

### Execution mode and MEM

MEM's history is a ring buffer that takes one entry per policy call. It only
matches training when it is fed once per environment step, so the adapter has
two modes, selected by `pi05_lerobot_step_mode`:

- `per_step` — every env step encodes a fresh observation and calls
  `select_action`, mirroring `lerobot-eval`. The policy replans on its own
  `n_action_steps` queue, so the additional calls are queue pops rather than
  extra forward passes.
- `chunk` — one action chunk per horizon, like the openpi adapter. Cheaper, but
  MEM then sees only one frame per horizon.

`auto` (the default) resolves to `per_step` for MEM checkpoints and `chunk`
otherwise. Measured with 4 env steps per horizon over 3 horizons:

| mode | env steps | observations MEM saw |
| ---- | --------- | -------------------- |
| `per_step` | 12 | 12 |
| `chunk` | 12 | 3 |

`pi05_lerobot_steps` (default 10) sets both the env steps per horizon and the
`n_action_steps` the worker overrides the checkpoint with. It matches the openpi
baseline in `policy/pi05`, where 50-step open-loop execution scored 0/19 on
`item_assembly`.

## Troubleshooting

**`PI0.5 worker exited unexpectedly`** — the worker's traceback is printed on
stderr just above. The two common causes are both checkpoint-portability issues:

*Stale config fields.* Older `config.json` files carry fields `PI05Config` has
since dropped (e.g. `optimizer_foreach`) and draccus refuses them. Strip
whatever is no longer a dataclass field:

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

*A tokenizer path from the training machine.* The saved preprocessor stores an
absolute `tokenizer_name`. Override it at eval time rather than editing the
checkpoint:

```bash
bash policy/pi05_lerobot/eval.sh click_bell random <ckpt> 0 \
    --pi05_lerobot_tokenizer google/paligemma-3b-pt-224
```

**Cameras mapped to the wrong sensor** — the adapter matches the checkpoint's
image feature names (`cam_high`/`base_0_rgb`/`camera1` … all understood) and
falls back to declaration order with a printed mapping. Check that line in the
eval log when a checkpoint uses unusual camera names.
