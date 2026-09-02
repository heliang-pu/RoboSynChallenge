# LiLa-WAM

[LiLa-WAM](https://github.com/teee000/LiLa-WAM) is a lightweight world-action model:
a frozen DINOv3 ViT-L/16 encoder plus a 0.2B trainable DiT action expert that
generates a 32-step action chunk by flow matching, with future-frame feature
prediction as an auxiliary objective. Tasks are conditioned through Visual
Transition Tokens (VTT) rather than language. 0.5B parameters in total, trainable
on a single 24 GB GPU.

The integration notes (design decisions, differences from upstream, FAQ) live in
[`policy/lila_wam/README_INTEGRATION.md`](https://github.com/EDEM-AI/RoboSynChallenge/blob/main/policy/lila_wam/README_INTEGRATION.md).

## Environment Setup

```bash
git submodule update --init policy/lila_wam/LiLa-WAM
bash policy/lila_wam/setup_env.sh --download-encoder   # training env + DINOv3 weights
uv pip install --python .venv/bin/python "transformers>=4.56,<5" omegaconf   # eval: the repo-root venv already has EmbodiChain
```

The DINOv3 checkpoint is gated on HuggingFace: accept the license at
<https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m> and run
`hf auth login` (or export `HF_TOKEN`) before `--download-encoder`.

## Generate RoboSynChallenge Data
See <a href="../collect_data.html">Collect Data Section</a> for more details.

## Prepare LiLa-WAM Data for Training

LiLa-WAM trains directly from a RoboSynChallenge **LeRobot v2.1** dataset — the
output of `bash launch/run_task.sh <task> <setting> 2_1 ...`. It reads
`observation.state` (14-dim joints), `action` (14-dim) and the camera video
streams. A v3.0 dataset must be converted first:

```bash
python scripts/convert_lerobot3.0_to_2.1.py --input <v3.0 dir> --output <v2.1 dir>
```

`finetune.sh` runs the preparation steps for you, but they are also available
individually:

```bash
python policy/lila_wam/build_frame_cache.py   --config <config.yaml>
python policy/lila_wam/compute_norm_stats.py  --config <config.yaml> --output <norm_stats.json>
python policy/lila_wam/precompute_task_cond.py --config <config.yaml>
```

The frame cache decodes each episode video once into JPEG buffers at the training
resolution; training samples frames in random order, and random access into AV1
video is far too slow for that.

For multi-task or sim+real co-training, list several roots in
`dataset.dataset_dir` / `dataset.task_names` of a config and pass that config to
`finetune.sh` instead of a dataset directory.

## Finetune model

```bash
# Stage 1 (lr 2e-4, upstream recommends stopping around epoch 11-12)
bash policy/lila_wam/finetune.sh \
     lerobot_dataset/click_bell/cobotmagic_Sim_click_bell click_bell 0 --epochs 12

# Stage 2 (lr 4e-5 for another 3-4 epochs; --init_from, NOT --resume, so the
# optimizer and lr schedule restart)
bash policy/lila_wam/finetune.sh \
     lerobot_dataset/click_bell/cobotmagic_Sim_click_bell click_bell 0 \
     --epochs 4 --learning_rate 4e-5 \
     --init_from policy/lila_wam/checkpoints/sft_<timestamp>/checkpoint_epoch_12.pt
```

Checkpoints land in `policy/lila_wam/checkpoints/sft_<timestamp>/`, together with
the `config.yaml` and `norm_stats.json` used for the run.

Two configs ship with the adapter: `configs/robosyn_3cam.yaml` (default; head +
both wrist cameras) and `configs/robosyn_cam_high.yaml` (single camera,
structurally identical to upstream).

## Eval on RoboSynChallenge

```bash
bash policy/lila_wam/eval.sh {task_name} {setting} {checkpoint} {model_name} {gpu_id}
```

Use one of these task names for `{task_name}`: `click_bell`, `handle_basket`,
`water_pouring`, `table_rearrangement`, `items_handover`, `drawer_open_place`,
`mixer_operating`, `item_assembly`, `manipulate_pipette`, `sample_loading`.

`{checkpoint}` can be a run directory (the highest epoch is picked automatically)
or a single `checkpoint_epoch_N.pt`. The VTT vector is looked up by task name;
override it with `--task_cond_name` when the training dataset was registered
under a different name.

The evaluation results, including videos, will be saved in the
`eval_result/{task_name}/lila_wam/{setting}/{train_config_name}/{model_name}/{checkpoint_id}/`
directory under the project root.
