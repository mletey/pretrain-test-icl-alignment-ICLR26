"""
Test run of training pipeline on data
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import sys
import matplotlib.pyplot as plt
import os

sys.path.append('../')
from data_generation.regression_structured import task_sampler, task_sampler_fixed_ws
from models.transformer import *
from models.train import *
import argparse


def unpack_sampler_batch(xte, yte):
    """
    xte: (B, L, D) where D=d+1 and last feature is y-slot
    yte: (B,) true y_{ℓ+1}

    Returns:
      x: (B, L, d)
      y: (B, L) with support y's filled and last token y=0 (from xte)
      d, l
    """
    B, L, D = xte.shape
    d = D - 1
    l = L - 1  # ℓ

    x = xte[..., :d]  # (B, L, d)

    # y in the input: support y's are stored in the last coordinate; query y is 0
    y = np.zeros((B, L), dtype=xte.dtype)
    y[:, :l] = xte[:, :l, -1]   # (B, ℓ)
    y[:, l] = xte[:, l, -1]     # should be 0

    return x, y, d, l


def construct_H_Z(x, y, l, d):
    # x: (B, L, d), y: (B, L)
    y_sum_x = np.einsum('nij,ni->nj', x[:, :l, :], y[:, :l])        # (B, d)
    y_sum_y = np.sum(y[:, :l] ** 2, axis=1)                         # (B,)
    H_Z = np.zeros((x.shape[0], d, d + 1), dtype=x.dtype)
    H_Z[:, :, :d] = x[:, l, :, None] * (d / l) * y_sum_x[:, None, :]  # (B, d, d)
    H_Z[:, :, d] = x[:, l] * (1 / l) * y_sum_y[:, None]               # (B, d)
    return H_Z


def compute_Gamma_star(n, d, H_Z, y_l1, lambda_val):
    H_Z_vec = H_Z.reshape(n, -1)
    regularization_term = (n / d) * lambda_val * np.eye(H_Z_vec.shape[1])
    sum_term = H_Z_vec.T @ H_Z_vec
    weighted_sum = H_Z_vec.T @ y_l1
    Gamma_star_vec = np.linalg.inv(regularization_term + sum_term) @ weighted_sum
    return Gamma_star_vec.reshape(d, d + 1)


def this_returns_Gammastar(x_train, y_train, *, lambda_val=1e-5):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train).reshape(-1)

    x, y, d, l = unpack_sampler_batch(x_train, y_train)
    n = x.shape[0]  # number of sequences in this batch

    H_Z = construct_H_Z(x, y, l, d)                     # (n, d, d+1)
    Gamma_star = compute_Gamma_star(n, d, H_Z, y_train, lambda_val)  # (d, d+1)

    # Prediction as inner product <H_Z, Gamma_star> for each sample
    y_hat = np.sum(H_Z * Gamma_star[None, :, :], axis=(1, 2))     # (n,)

    return Gamma_star

def this_returns_theory_predictions(x_test, y_test, Gamma_star):
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    x, y, d, l = unpack_sampler_batch(x_test, y_test)
    H_Z = construct_H_Z(x, y, l, d)                     # (n, d, d+1)  
    y_hat = np.sum(H_Z * Gamma_star[None, :, :], axis=(1, 2))     # (n,)
    return y_hat


def parse_args():
    p = argparse.ArgumentParser()

    # --- data ---
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--rho", type=float, default=0.01)

    # --- model ---
    p.add_argument("--n_hidden", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_mlp_layers", type=int, default=1)  
    p.add_argument("--mlp_multiplier", type=int, default=2)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--pure_linear_self_att", action="store_true")
    p.add_argument("--use_input_projection", action="store_true", default=True)
    p.add_argument("--no_input_projection", action="store_true", help="If set, skip input projection (hidden=D).")

    # --- training ---
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=10_000)

    p.add_argument("--base_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--end_lr", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)

    p.add_argument("--log_every_steps", type=int, default=100)
    p.add_argument("--full_train_eval_every_steps", type=int, default=2000)
    p.add_argument("--full_train_eval_batch_size", type=int, default=1024)
    p.add_argument("--test_eval_every_steps", type=int, default=1000)
    p.add_argument("--test_eval_batch_size", type=int, default=None)

    p.add_argument("--best_ckpt_path", type=str, default=None)

    p.add_argument("--plot_name", type=str, default="pred_v_true")

    # --- wandb ---
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None,
                   help="Comma-separated tags, e.g. 'sweep,h100,debug'")

    args = p.parse_args()

    # reconcile projection flags
    if args.no_input_projection:
        args.use_input_projection = False

    # parse tags
    if args.wandb_tags is not None:
        args.wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    return args


def main():
    args = parse_args()

    l = int(args.alpha * args.d);
    n = int(args.tau * (args.d**2));
    if args.kappa < 0:
        k = -1
        print("using k = ", k, flush=True)
    else:
        k = int(args.kappa * (args.d));
    h = args.d+1;

    C = np.eye(args.d)
    trainobject = task_sampler(args.d, l, n, k, args.rho, C, 1)
    testobject = task_sampler(args.d, l, n, -1, args.rho, C, 1)

    # --- model config ---
    config = TransformerConfig(
        n_hidden=args.d + 1,
        n_layers=args.n_layers,
        n_mlp_layers=args.n_mlp_layers,
        mlp_multiplier=args.mlp_multiplier,
        dropout_rate=args.dropout_rate,
        pure_linear_self_att=args.pure_linear_self_att,
        use_input_projection=args.use_input_projection,
    )

    # --- train ---
    state, hist, best, train_x, train_y, train_w = train(
        config,
        data_iter=iter(trainobject),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        end_lr=args.end_lr,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed if args.seed != 0 else None,
        log_every_steps=args.log_every_steps,
        full_train_eval_every_steps=args.full_train_eval_every_steps,
        full_train_eval_batch_size=args.full_train_eval_batch_size,
        test_iter=iter(testobject),  
        test_eval_every_steps=args.test_eval_every_steps,
        test_eval_batch_size=args.test_eval_batch_size,
        best_ckpt_path=args.best_ckpt_path,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
    )

    xte, yte, _ = next(testobject)
    preds = state.apply_fn({"params": state.params}, jnp.asarray(xte), deterministic=True)
    y_true = np.asarray(yte).reshape(-1)
    y_pred = np.asarray(jax.device_get(preds)).reshape(-1)
    plt.scatter(y_true, y_pred, color='red', alpha=0.5, s=2)

    # Gamma = this_returns_Gammastar(train_x, train_y, lambda_val=1e-5)
    # y_pred_theory = this_returns_theory_predictions(xte,yte,Gamma)
    # plt.scatter(y_true, y_pred_theory, color='green', alpha=0.5, s=2)

    idg_object = task_sampler_fixed_ws(args.d, l, n, train_w, args.rho, 1)
    xte, yte = next(idg_object)
    preds = state.apply_fn({"params": state.params}, jnp.asarray(xte), deterministic=True)
    y_true_idg = np.asarray(yte).reshape(-1)
    y_pred_idg = np.asarray(jax.device_get(preds)).reshape(-1)
    plt.scatter(y_true_idg, y_pred_idg, color='blue', alpha=0.4, s=2)

    fig_path = f'plots/{args.plot_name}.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # save CSV
    csv_path = 'pred_v_true.csv'
    np.savetxt(
        csv_path,
        np.column_stack([y_true, y_pred]),
        delimiter=",",
        header="y_true,y_pred",
        comments="",
    )


if __name__ == "__main__":
    main()
    print("fml")
