from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def make_test_m_matrix(csv_path: str | Path):
    csv_path = Path(csv_path)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Optional: sanity check expected columns exist
    required = {"kappa", "train_power", "test_m", "seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    # Ensure numeric types (in case they were read as strings)
    for col in ["kappa", "train_power", "test_m", "seed"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Average over seeds for each (kappa, train_power)
    grouped = (
        df.groupby(["kappa", "train_power"], as_index=False)["test_m"]
          .mean()
          .rename(columns={"test_m": "test_m_mean"})
    )

    # Define consistent ordering for axes
    kappas = np.sort(df["kappa"].unique())
    train_powers = np.sort(df["train_power"].unique())

    # Pivot into matrix (rows=kappa, cols=train_power)
    mat_df = grouped.pivot(index="kappa", columns="train_power", values="test_m_mean")

    # Reindex to enforce full rectangular shape in the assumed ordering
    mat_df = mat_df.reindex(index=kappas, columns=train_powers)

    # If your assumption "all combos exist" is true, there should be no NaNs
    if mat_df.isna().any().any():
        missing_pairs = mat_df.isna().stack()
        missing_pairs = missing_pairs[missing_pairs].index.tolist()
        raise ValueError(
            "Some (kappa, train_power) combinations are missing in the data. "
            f"Missing pairs (showing up to 20): {missing_pairs[:20]}"
        )

    # Convert to numpy matrix
    test_m_matrix = mat_df.to_numpy()

    return kappas, train_powers, test_m_matrix


if __name__ == "__main__":

    CSV_PATH = "/n/netscratch/pehlevan_lab/Lab/ml/ICLR26_finalruns/test_results/change_train_power_d32/results_csv"
    kappas, train_powers, M = make_test_m_matrix(CSV_PATH)

    test_power = 0.9 
    j = np.where(train_powers == test_power)[0][0] 
    v = M[:, j]                             

    percent_change = np.zeros(M.shape)
    for col in range(len(train_powers)):
        percent_change[:,col] = -100*(M[:,col] - v)/v
    
    sns.set(style="white",font_scale=1.5,palette="colorblind")
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams["figure.figsize"] = (24, 20)

    sample = 15
    cyans = LinearSegmentedColormap.from_list("what", ["#E8FDFB","#40E0D0", "#0AA192"])
    colors = np.vstack(
        (
            plt.get_cmap("Oranges", sample)(np.linspace(0, 1, sample))[::-1],
            np.ones((1, 4)),
            cyans(np.linspace(0, 2, sample)),
        )
    )
    cmap_mary = LinearSegmentedColormap.from_list("green_to_red", colors)

    rows = train_powers - test_power
    columns = kappas

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(percent_change.T,
            yticklabels=np.linspace(0, len(rows) - 1, 12, dtype=int),
            xticklabels=np.linspace(0, len(columns) - 1, 10, dtype=int),  
            center=0,
            cmap = cmap_mary,
            annot=False,
            linewidths=0.5,
            cbar=True,
            ax=ax,
            cbar_kws={'label': '% Improvement in Error'})
    
    # KAPPA, DELTA = np.meshgrid(np.arange(len(columns)), np.arange(len(rows)))
    # contours = ax.contour(KAPPA, DELTA, percent_change.T, levels=[0], colors='black', linewidths=1.2)
    # labels = ax.clabel(contours, fmt='%1.0f', fontsize=13, inline=True)
    # for txt in labels:
    #     txt.set_rotation(0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)

    ax.set_xticks(np.linspace(0, len(columns) - 1, 10, dtype=int))
    ax.set_xticklabels([f"{val:.1f}" for val in columns[np.linspace(0, len(columns) - 1, 10, dtype=int)]], rotation = 90, fontsize=13)


    ax.set_yticks(np.linspace(2, len(rows)-1, 7, dtype=int))
    ax.set_yticklabels([f"{val:.1f}" for val in rows[np.linspace(2, len(rows)-1, 7, dtype=int)]], fontsize=13)
    ax.invert_yaxis()

    ax.set_xlabel(r"$\kappa$ = $k/d$")
    ax.set_ylabel(r"Train power $-$ Test power")

    fig.savefig('FINALLY.png', dpi=200)


    