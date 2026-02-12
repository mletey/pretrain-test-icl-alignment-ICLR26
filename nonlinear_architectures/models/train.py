"""
Transformer training pipeline

Data inputs:
x has shape (B, L, D), y has shape (B,)
Offline training

Mary Letey
January 2026
With help from ChatGPT 5.2 Thinking
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax import serialization


Batch = Tuple[np.ndarray, np.ndarray]  # (xs, ys) on host


def new_seed() -> int:
    return int(np.random.randint(1, np.iinfo(np.int32).max))


def mse_loss(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(preds - targets))


def mae_loss(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(preds - targets))


def make_lr_schedule(
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    end_lr: float = 0.0,
) -> optax.Schedule:
    """Warmup + cosine decay schedule."""
    warmup_steps = int(max(0, warmup_steps))
    total_steps = int(max(1, total_steps))
    decay_steps = int(max(1, total_steps - warmup_steps))

    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps if warmup_steps > 0 else 1,
    )
    cosine = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=decay_steps,
        alpha=(end_lr / base_lr) if base_lr > 0 else 0.0,
    )
    return optax.join_schedules([warmup, cosine], boundaries=[warmup_steps])


class TrainState(train_state.TrainState):
    pass


def create_train_state(
    rng: jax.Array,
    model,
    dummy_x: np.ndarray,
    *,
    lr_schedule: optax.Schedule,
    weight_decay: float = 0.0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    grad_clip_norm: Optional[float] = 1.0,
) -> TrainState:
    variables = model.init(rng, jnp.asarray(dummy_x), deterministic=True)
    params = variables["params"]

    tx_parts = []
    if grad_clip_norm is not None:
        tx_parts.append(optax.clip_by_global_norm(grad_clip_norm))

    tx_parts.append(
        optax.adamw(
            learning_rate=lr_schedule,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
        )
    )

    tx = optax.chain(*tx_parts)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.Array,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    x, y = batch  # x: (B,L,D), y: (B,)

    def loss_fn(params):
        preds = state.apply_fn(
            {"params": params},
            x,
            deterministic=False,
            rngs={"dropout": rng},
        )  # (B,)
        loss = mse_loss(preds, y)
        metrics = {
            "loss": loss,
            "mse": loss,
            "mae": mae_loss(preds, y),
            "pred_mean": jnp.mean(preds),
            "pred_std": jnp.std(preds),
            "y_mean": jnp.mean(y),
            "y_std": jnp.std(y),
        }
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics

@jax.jit
def eval_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    x, y = batch
    preds = state.apply_fn({"params": state.params}, x, deterministic=True)
    return {
        "mse": mse_loss(preds, y),
        "mae": mae_loss(preds, y),
        "pred_mean": jnp.mean(preds),
        "pred_std": jnp.std(preds),
        "y_mean": jnp.mean(y),
        "y_std": jnp.std(y),
    }

def iterate_minibatches(
    rng: np.random.Generator,
    xs: np.ndarray,
    ys: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
) -> Iterator[Batch]:
    n = xs.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)

    for start in range(0, n, batch_size):
        sl = idx[start : start + batch_size]
        yield xs[sl], ys[sl]


def eval_dataset_minibatched(
    state: TrainState,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    batch_size: int,
) -> Dict[str, float]:
    n = xs.shape[0]
    sums = {k: 0.0 for k in ["mse", "mae", "pred_mean", "pred_std", "y_mean", "y_std"]}
    count = 0

    for start in range(0, n, batch_size):
        xb = jnp.asarray(xs[start : start + batch_size])
        yb = jnp.asarray(ys[start : start + batch_size])
        m = eval_step(state, (xb, yb))
        bsz = xb.shape[0]
        for k in sums:
            sums[k] += float(m[k]) * bsz
        count += bsz

    return {k: sums[k] / count for k in sums}


def save_params_checkpoint(params: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))


@dataclass
class TrainHistory:
    train: Dict[str, list]
    eval_full_train: Dict[str, list]
    eval_test: Dict[str, list]


def train(
    config,
    data_iter: Iterator[Batch],
    *,
    batch_size: int,
    num_epochs: int = 10,
    base_lr: float = 1e-4,
    weight_decay: float = 0.0,
    warmup_steps: int = 500,
    end_lr: float = 0.0,
    grad_clip_norm: Optional[float] = 1.0,
    seed: Optional[int] = None,
    log_every_steps: int = 100,
    full_train_eval_every_steps: Optional[int] = 2000,
    full_train_eval_batch_size: int = 1024,
    test_iter: Optional[Iterator[Batch]] = None,
    test_eval_every_steps: Optional[int] = 1000,
    test_eval_batch_size: Optional[int] = None,
    best_ckpt_path: Optional[str] = None,
    # W&B
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_tags: Optional[list] = None,
) -> Tuple[TrainState, TrainHistory, Optional[Any]]:
    """
    Returns: (final_state, history, best_params)

    Best params are selected by lowest test MSE from sampled test batches (test_iter).
    If best_ckpt_path is provided, best params are saved there when they improve.
    """
    if seed is None:
        seed = new_seed()

    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            tags=wandb_tags,
            config={
                "seed": seed,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "base_lr": base_lr,
                "end_lr": end_lr,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "grad_clip_norm": grad_clip_norm,
                "log_every_steps": log_every_steps,
                "full_train_eval_every_steps": full_train_eval_every_steps,
                "test_eval_every_steps": test_eval_every_steps,
                "best_ckpt_path": best_ckpt_path,
                "model_config": dict(config.__dict__) if hasattr(config, "__dict__") else str(config),
            },
        )

    # Fixed training dataset sample
    xs, ys, ws = next(data_iter)
    assert xs.ndim == 3, f"expected xs.ndim=3, got {xs.shape}"
    assert ys.ndim == 1, f"expected ys.ndim=1, got {ys.shape}"
    assert xs.shape[0] == ys.shape[0], "xs and ys batch dimension mismatch"

    n_data = xs.shape[0]
    dummy_x = xs[: min(batch_size, n_data)]

    # Steps count for schedule (epochs * steps_per_epoch)
    steps_per_epoch = int(np.ceil(n_data / batch_size))
    total_steps = int(num_epochs * steps_per_epoch)

    # lr_schedule = make_lr_schedule(
    #     base_lr=base_lr,
    #     warmup_steps=warmup_steps,
    #     total_steps=total_steps,
    #     end_lr=end_lr,
    # )
    lr_schedule = optax.constant_schedule(base_lr)

    # init model/state
    model = config.to_model()
    jax_rng = jax.random.key(seed)
    state = create_train_state(
        jax_rng,
        model,
        dummy_x,
        lr_schedule=lr_schedule,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
    )

    np_rng = np.random.default_rng(seed)

    history = TrainHistory(
        train={k: [] for k in ["loss", "mse", "mae", "pred_mean", "pred_std", "y_mean", "y_std", "lr"]},
        eval_full_train={k: [] for k in ["mse", "mae", "pred_mean", "pred_std", "y_mean", "y_std"]},
        eval_test={k: [] for k in ["mse", "mae", "pred_mean", "pred_std", "y_mean", "y_std"]},
    )

    best_params = None
    best_test_mse = np.inf

    global_step = 0

    for epoch in range(num_epochs):
        for xb, yb in iterate_minibatches(np_rng, xs, ys, batch_size, shuffle=True):
            global_step += 1

            xb_j = jnp.asarray(xb)
            yb_j = jnp.asarray(yb)

            # per-step dropout rng
            jax_rng, step_rng = jax.random.split(jax_rng)

            state, metrics = train_step(state, (xb_j, yb_j), step_rng)

            lr_now = float(lr_schedule(global_step - 1))

            if global_step % log_every_steps == 0:
                m_host = {k: float(v) for k, v in metrics.items()}
                m_host["lr"] = lr_now

                for k, v in m_host.items():
                    if k in history.train:
                        history.train[k].append(v)

                if use_wandb:
                    import wandb
                    wandb.log({f"train/{k}": v for k, v in m_host.items()}, step=global_step)

            # Full training-set eval (sanity)
            if full_train_eval_every_steps is not None and global_step % full_train_eval_every_steps == 0:
                fm = eval_dataset_minibatched(state, xs, ys, batch_size=full_train_eval_batch_size)
                for k, v in fm.items():
                    history.eval_full_train[k].append(v)

                if use_wandb:
                    import wandb
                    wandb.log({f"eval_full_train/{k}": v for k, v in fm.items()}, step=global_step)

                print(
                    f"[step {global_step:>7}] FULL_TRAIN mse={fm['mse']:.6f} mae={fm['mae']:.6f} lr={lr_now:.2e}"
                )

            # Test eval + best checkpoint
            if test_iter is not None and test_eval_every_steps is not None and global_step % test_eval_every_steps == 0:
                xte, yte, _ = next(test_iter)
                assert xte.ndim == 3 and yte.ndim == 1

                if test_eval_batch_size is None or xte.shape[0] <= test_eval_batch_size:
                    tm = eval_step(state, (jnp.asarray(xte), jnp.asarray(yte)))
                    tm = {k: float(v) for k, v in tm.items()}
                else:
                    tm = eval_dataset_minibatched(state, xs, ys, batch_size=test_eval_batch_size)

                for k, v in tm.items():
                    history.eval_test[k].append(v)

                if use_wandb:
                    import wandb
                    wandb.log({f"eval_test/{k}": v for k, v in tm.items()}, step=global_step)

                print(
                    f"[step {global_step:>7}] TEST_EVAL mse={tm['mse']:.6f} mae={tm['mae']:.6f} "
                    f"(best {best_test_mse:.6f}) lr={lr_now:.2e}"
                )

                if tm["mse"] < best_test_mse:
                    best_test_mse = tm["mse"]
                    best_params = jax.device_get(state.params)

                    if use_wandb:
                        import wandb
                        wandb.summary["best_test_mse"] = best_test_mse
                        wandb.summary["best_step"] = global_step

                    if best_ckpt_path is not None:
                        save_params_checkpoint(best_params, best_ckpt_path)
                        print(f"  saved new best params to: {best_ckpt_path}")

    if use_wandb:
        import wandb
        wandb.summary["final_step"] = global_step
        if np.isfinite(best_test_mse):
            wandb.summary["best_test_mse"] = best_test_mse
        wandb.finish()

    return state, history, best_params, xs, ys, ws