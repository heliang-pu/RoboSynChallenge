#!/usr/bin/env python3
"""Run the standard JAX trainer with a verified effective batch size.

``TrainConfig.batch_size`` remains the physical micro batch.  The optimizer is
wrapped in :class:`optax.MultiSteps`, so parameters are updated only after
``OPENPI_GRADIENT_ACCUMULATION_STEPS`` micro batches.  The inner Adam/LR state
therefore advances once per effective batch, while ``TrainState.step`` keeps
counting micro batches so diffusion RNGs stay unique and Orbax resume remains
exact.
"""

from __future__ import annotations

import os

import jax
import optax

import train as _train


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def main() -> None:
    accumulation_steps = _positive_int_env("OPENPI_GRADIENT_ACCUMULATION_STEPS", 1)
    expected_effective_batch = _positive_int_env("OPENPI_EFFECTIVE_BATCH_SIZE", 56)
    save_every_updates = _positive_int_env("OPENPI_SAVE_EVERY_UPDATES", 1)
    first_save_update = _positive_int_env("OPENPI_FIRST_SAVE_UPDATE", save_every_updates)
    smoke_no_checkpoint = _env_flag("OPENPI_SMOKE_NO_CHECKPOINT")
    config = _train._config.cli()

    effective_batch = config.batch_size * accumulation_steps
    if effective_batch != expected_effective_batch:
        raise ValueError(
            f"micro_batch={config.batch_size} * accumulation={accumulation_steps} "
            f"= {effective_batch}, expected effective batch {expected_effective_batch}"
        )
    if config.num_train_steps % accumulation_steps:
        raise ValueError(
            f"num_train_steps={config.num_train_steps} must be divisible by "
            f"accumulation_steps={accumulation_steps}"
        )
    if config.ema_decay is not None and accumulation_steps != 1:
        raise ValueError("EMA must be disabled when gradient accumulation is enabled")
    if accumulation_steps != 1 and config.save_interval != 1:
        raise ValueError(
            "pass --save-interval=1; train_accum filters those callbacks to "
            "optimizer-update boundaries"
        )

    total_updates = config.num_train_steps // accumulation_steps
    base_create_optimizer = _train._optimizer.create_optimizer
    base_save_state = _train._checkpoints.save_state

    def create_accumulating_optimizer(optimizer, lr_schedule, weight_decay_mask=None):
        base = base_create_optimizer(optimizer, lr_schedule, weight_decay_mask)
        if accumulation_steps == 1:
            return base
        # Accumulate/average first, then run the existing
        # clip_by_global_norm -> AdamW chain once per effective batch.
        return optax.MultiSteps(
            base,
            every_k_schedule=accumulation_steps,
            use_grad_mean=True,
        ).gradient_transformation()

    def save_only_on_update_boundary(manager, state, data_loader, step, *args, **kwargs):
        if accumulation_steps == 1:
            return base_save_state(manager, state, data_loader, step, *args, **kwargs)
        # The stock loop calls save_state after the update.  Its zero-based
        # ``step`` therefore corresponds to ``state.step == step + 1``.
        # Use that host integer for the common fast path; avoid synchronizing
        # two device scalars on every micro step.
        completed_micro_steps = int(step) + 1
        if completed_micro_steps % accumulation_steps:
            return False
        gradient_step = completed_micro_steps // accumulation_steps
        is_requested = (
            gradient_step == first_save_update
            or gradient_step % save_every_updates == 0
        )
        is_final = gradient_step == total_updates
        if not (is_requested or is_final):
            return False
        mini_step = int(jax.device_get(state.opt_state.mini_step))
        optimizer_gradient_step = int(jax.device_get(state.opt_state.gradient_step))
        if mini_step != 0 or optimizer_gradient_step != gradient_step:
            raise RuntimeError(
                "optimizer accumulation state disagrees with TrainState.step: "
                f"mini_step={mini_step}, optimizer_gradient_step={optimizer_gradient_step}, "
                f"expected_gradient_step={gradient_step}"
            )
        print(
            "[gradient-accumulation] checkpoint boundary "
            f"micro_step={completed_micro_steps} optimizer_update={gradient_step}/{total_updates}",
            flush=True,
        )
        if smoke_no_checkpoint:
            return False
        return base_save_state(manager, state, data_loader, step, *args, **kwargs)

    _train._optimizer.create_optimizer = create_accumulating_optimizer
    _train._checkpoints.save_state = save_only_on_update_boundary
    print(
        "[gradient-accumulation] "
        f"micro_batch={config.batch_size} accumulation_steps={accumulation_steps} "
        f"effective_batch={effective_batch} micro_steps={config.num_train_steps} "
        f"optimizer_updates={total_updates} first_save_update={first_save_update} "
        f"save_every_updates={save_every_updates}",
        flush=True,
    )
    _train.main(config)


if __name__ == "__main__":
    main()
