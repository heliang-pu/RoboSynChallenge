#!/usr/bin/env python3
"""Run the standard JAX trainer with a verified effective batch size.

``TrainConfig.batch_size`` remains the physical micro batch.  The optimizer is
wrapped in :class:`optax.MultiSteps`, so parameters are updated only after
``OPENPI_GRADIENT_ACCUMULATION_STEPS`` micro batches.  The inner Adam/LR state
therefore advances once per effective batch, while ``TrainState.step`` keeps
counting micro batches so diffusion RNGs stay unique. Orbax restores the
model/optimizer/accumulator state; the stock shuffled data iterator itself is
not resumable.
"""

from __future__ import annotations

import dataclasses
import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.shared.nnx_utils as nnx_utils

import train as _train


class _BFloat16MultiSteps:
    """Low-memory gradient averaging with a bfloat16 accumulator tree.

    Model parameters and the wrapped Adam state keep their original dtypes.
    Only gradients waiting for the next effective-batch update are stored in
    bfloat16.  On an update boundary the bfloat16 mean is passed into clipping
    and AdamW; optimizer moment arithmetic is promoted by its float32 state.
    This deliberately avoids materializing another full float32 gradient tree,
    which does not fit on a 64GB DCU.
    """

    def __init__(self, opt: optax.GradientTransformation, every_k: int):
        self._opt = optax.with_extra_args_support(opt)
        self._every_k = every_k

    @staticmethod
    def _zero_accumulator(value):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.inexact):
            return jnp.zeros(value.shape, dtype=jnp.bfloat16)
        return jnp.zeros_like(value)

    def init(self, params):
        return optax.MultiStepsState(
            mini_step=jnp.zeros([], dtype=jnp.int32),
            gradient_step=jnp.zeros([], dtype=jnp.int32),
            inner_opt_state=self._opt.init(params),
            acc_grads=jax.tree.map(self._zero_accumulator, params),
            skip_state=(),
        )

    def update(self, updates, state, params=None, **extra_args):
        divisor = (state.mini_step + 1).astype(jnp.bfloat16)

        def accumulate(grad, acc):
            if not (hasattr(grad, "dtype") and jnp.issubdtype(grad.dtype, jnp.inexact)):
                return acc
            grad_bf16 = grad.astype(jnp.bfloat16)
            return acc + (grad_bf16 - acc) / divisor

        accumulated = jax.tree.map(accumulate, updates, state.acc_grads)
        emit = state.mini_step == self._every_k - 1

        def emit_update(operand):
            current_updates, current_acc, current_state, current_params = operand
            final_updates, new_inner_state = self._opt.update(
                current_acc,
                current_state.inner_opt_state,
                params=current_params,
                **extra_args,
            )
            new_state = optax.MultiStepsState(
                mini_step=jnp.zeros_like(current_state.mini_step),
                gradient_step=current_state.gradient_step + jnp.ones_like(current_state.gradient_step),
                inner_opt_state=new_inner_state,
                acc_grads=jax.tree.map(jnp.zeros_like, current_state.acc_grads),
                skip_state=current_state.skip_state,
            )
            return final_updates, new_state

        def hold_update(operand):
            current_updates, current_acc, current_state, _ = operand
            new_state = optax.MultiStepsState(
                mini_step=current_state.mini_step + jnp.ones_like(current_state.mini_step),
                gradient_step=current_state.gradient_step,
                inner_opt_state=current_state.inner_opt_state,
                acc_grads=current_acc,
                skip_state=current_state.skip_state,
            )
            return jax.tree.map(jnp.zeros_like, current_updates), new_state

        return jax.lax.cond(
            emit,
            emit_update,
            hold_update,
            (updates, accumulated, state, params),
        )

    def gradient_transformation(self) -> optax.GradientTransformation:
        return optax.GradientTransformation(init=self.init, update=self.update)


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
    accumulator_dtype = os.environ.get("OPENPI_ACCUMULATOR_DTYPE", "float32").strip().lower()
    if accumulator_dtype not in {"float32", "bfloat16"}:
        raise ValueError(
            f"OPENPI_ACCUMULATOR_DTYPE must be float32 or bfloat16, got {accumulator_dtype!r}"
        )
    config = _train._config.cli()
    freeze_mode = os.environ.get("OPENPI_FREEZE_MODE", "none").strip().lower()
    if freeze_mode == "vlm":
        # Preserve the pretrained visual/language representation and specialize
        # the action expert plus the action/time projection layers per task.
        action_expert = nnx_utils.PathRegex(".*llm.*_1.*")
        freeze_filter = nnx.Any(
            nnx_utils.PathRegex(".*img.*"),
            nnx.All(
                nnx_utils.PathRegex(".*llm.*"),
                nnx.Not(action_expert),
            ),
        )
        config = dataclasses.replace(config, freeze_filter=freeze_filter)
    elif freeze_mode != "none":
        raise ValueError(f"OPENPI_FREEZE_MODE must be none or vlm, got {freeze_mode!r}")

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
        if accumulator_dtype == "bfloat16":
            return _BFloat16MultiSteps(base, accumulation_steps).gradient_transformation()
        return optax.MultiSteps(base, every_k_schedule=accumulation_steps, use_grad_mean=True).gradient_transformation()

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
        f"accumulator_dtype={accumulator_dtype} "
        f"freeze_mode={freeze_mode} "
        f"effective_batch={effective_batch} micro_steps={config.num_train_steps} "
        f"optimizer_updates={total_updates} first_save_update={first_save_update} "
        f"save_every_updates={save_every_updates}",
        flush=True,
    )
    _train.main(config)


if __name__ == "__main__":
    main()
