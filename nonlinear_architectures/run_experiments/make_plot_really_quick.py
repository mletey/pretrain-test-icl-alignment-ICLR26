import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from theory.theory import ICL_error, icl_error_finite

import itertools

def main():
    csvpath= '/n/netscratch/pehlevan_lab/Lab/ml/ICLR26_finalruns/test_results/linearattention_power0p9_d16/results_csv'
    df = pd.read_csv(csvpath)

    # Keep what we need for plotting + the group-constants you want to access
    cols = ["kappa", "test_power", "test_m", "test_s", "alpha", "tau", "train_power", "d"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[cols].copy()

    # Coerce numeric
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)

    # Aggregate duplicates while carrying alpha/tau through
    df = (
        df.groupby(["test_power", "kappa"], as_index=False)
        .agg(
            test_m=("test_m", "mean"),
            test_s=("test_s", "mean"),
            alpha=("alpha", "first"),
            tau=("tau", "first"),
            train_power=("train_power", "first"),
            d=("d", "first"),
        )
    )

    fig, ax = plt.subplots()

    COLOR_LIST = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
        "black",
    ]
    color_cycle = itertools.cycle(COLOR_LIST)


    for tp, g in df.groupby("test_power"):
        print(tp)

        c = next(color_cycle)

        g = g.sort_values("kappa")
        x = g["kappa"].to_numpy()
        m = g["test_m"].to_numpy()
        s = g["test_s"].to_numpy()

        ax.scatter(x, m, label=f"test_power={tp:g}", s=10, color=c)
        ax.fill_between(x, m - s, m + s, alpha=0.1, color=c)

        train_power = g["train_power"].iloc[0]
        d = g["d"].iloc[0]
        alpha = g["alpha"].iloc[0]
        tau   = g["tau"].iloc[0]
        Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d
        Ctest = np.diag(np.array([(j + 1) ** -tp for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
        
        
    Ctr = np.diag(np.array([(j + 1) ** -0.9 for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d
    finite_vals = [icl_error_finite(d, tau, alpha, kappa, Ctr, Ctr, 5) for kappa in x[1:]]
    #theory_vals = [ICL_error(Ctr, Ctr, tau, alpha, kappa, 0.01, numavg=50) for kappa in x]
    ax.plot(x[1:], finite_vals, color='black', label='SIMULATION TRAIN!')

    ax.set_xlabel("kappa")
    ax.set_ylabel("test_m")
    ax.set_title("test_m vs kappa (shaded ± test_s)")
    ax.legend(title="Curves")

    ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig('testplot-theory.png', dpi=200)

if __name__ == "__main__":
    main()