"""
Trainable linear attention (powerlaw) -- Recreating figure 1
"""

import numpy as np
import optax
import sys
import os
import csv

sys.path.append('../')
from data_generation.regression_structured import task_sampler
from models.transformer import *
from models.train import *
import argparse

def parse_args():
    p = argparse.ArgumentParser()

    # --- data ---
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--task_power", type=float, default=0.0)

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

    # --- wandb ---
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None,
                   help="Comma-separated tags, e.g. 'sweep,h100,debug'")
    
    p.add_argument("--savedirectory", type=str, default='./temp.csv')

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
    else:
        k = int(args.kappa * (args.d));
    h = args.d+1;

    C = np.diag(np.array([(j + 1) ** -args.task_power for j in range(args.d)])); C = (C/np.trace(C))*args.d

    trainobject = task_sampler(args.d, l, n, k, args.rho, C, 1)
    testobject = task_sampler(args.d, l, n, -1, args.rho, C, 1)

    # --- model config ---
    config = TransformerConfig(
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_mlp_layers=args.n_mlp_layers,
        mlp_multiplier=args.mlp_multiplier,
        dropout_rate=args.dropout_rate,
        pure_linear_self_att=args.pure_linear_self_att,
        use_input_projection=args.use_input_projection,
    )

    # --- train ---
    state, hist, best, train_x, train_y, train_w  = train(
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
        test_iter=iter(testobject),  # replace with iter(testobject) when you have one
        test_eval_every_steps=args.test_eval_every_steps,
        test_eval_batch_size=args.test_eval_batch_size,
        best_ckpt_path=args.best_ckpt_path,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
    )
    print("FINISHED TRAINING!", flush=True)
    powers = np.linspace(max(0,args.task_power-0.5), args.task_power+0.5, 11)

    numsamples = 500
    CSV_COLUMNS = ["d", "kappa", "alpha", "tau", "train_power", "test_power", "test_m", "test_s"]
    loss_func = optax.squared_error

    for test_power in powers:
        print(f"starting test on power {test_power} ...")
        Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(args.d)])); Ctest = (Ctest/np.trace(Ctest))*args.d
        testobject = task_sampler(args.d, l, n, -1, args.rho, Ctest, 1)
        tracker = []
        for _ in range(numsamples):
            xs, labels, _ = next(testobject); # generates data
            logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
            tracker.append(loss_func(logits, labels).mean())
        tracker = np.array(tracker)
        test_m = np.mean(tracker)
        test_s = np.std(tracker)

        append_row_csv(args.savedirectory, {
            "d": args.d,
            "kappa": args.kappa,
            "alpha": args.alpha,
            "tau": args.tau,
            "train_power": args.task_power,
            "test_power": float(test_power),
            "test_m": test_m,
            "test_s": test_s,
        }, CSV_COLUMNS)
        print(f"... finished test on power {test_power}")


def append_row_csv(path: str, row: dict, CSV_COLUMNS):
    # Ensure parent dir exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create header only if file missing or empty
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if need_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()  # flush each iteration so you don't lose data on preemption
        os.fsync(f.fileno())  # force write to disk (slower but safer)

if __name__ == "__main__":
    main()
    print("fml")

