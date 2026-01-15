import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _hdf5_paths(data_root: str, date: str):
    base = Path(data_root) / date
    return base / "data_train.hdf5", base / "data_val.hdf5"


def _trial_keys(f: h5py.File):
    return sorted([k for k in f.keys() if k.startswith("trial_")])


def _ensure_outdir():
    outdir = Path("figures") / "phase1"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _save_fig(fig, outdir: Path, stem: str):
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def _choose_sample(trial_keys, max_trials: int, seed: int = 0):
    keys = list(trial_keys)
    if len(keys) <= max_trials:
        return keys
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(keys), size=max_trials, replace=False)
    idx = np.sort(idx)
    return [keys[i] for i in idx]


def main():
    parser = argparse.ArgumentParser(description="Phase 1: neural feature statistics and plots")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()

    train_path, val_path = _hdf5_paths(args.data_root, args.date)
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val file: {val_path}")

    outdir = _ensure_outdir()

    # Build a unified pool of trial keys across train+val
    with h5py.File(train_path, "r") as f_tr, h5py.File(val_path, "r") as f_va:
        keys_tr = [("train", k) for k in _trial_keys(f_tr)]
        keys_va = [("val", k) for k in _trial_keys(f_va)]

    pool = keys_tr + keys_va
    sample = _choose_sample(pool, max_trials=200, seed=0)
    print(f"Sampling {len(sample)} trials (seed=0) across train+val")

    # Online per-channel mean/var across all sampled timepoints
    sum_x = np.zeros(512, dtype=np.float64)
    sum_x2 = np.zeros(512, dtype=np.float64)
    n_total = 0

    # Correlation matrix: collect a limited number of timepoints total
    corr_timepoints_per_trial = 25
    corr_rows = []

    # Temporal energy curve: online mean/std with variable lengths
    energy_sum = np.zeros(1, dtype=np.float64)
    energy_sum2 = np.zeros(1, dtype=np.float64)
    energy_count = np.zeros(1, dtype=np.int64)

    rng = np.random.default_rng(0)

    def _accum_energy(e: np.ndarray):
        nonlocal energy_sum, energy_sum2, energy_count
        T = e.shape[0]
        if T > energy_sum.shape[0]:
            newT = T
            energy_sum = np.pad(energy_sum, (0, newT - energy_sum.shape[0]), constant_values=0.0)
            energy_sum2 = np.pad(energy_sum2, (0, newT - energy_sum2.shape[0]), constant_values=0.0)
            energy_count = np.pad(energy_count, (0, newT - energy_count.shape[0]), constant_values=0)
        energy_sum[:T] += e
        energy_sum2[:T] += e * e
        energy_count[:T] += 1

    # Iterate sampled trials without loading full dataset into RAM
    for split, tk in sample:
        h5_path = train_path if split == "train" else val_path
        with h5py.File(h5_path, "r") as f:
            x = f[tk]["input_features"]  # (T, 512)
            T, C = x.shape
            if C != 512:
                raise RuntimeError(f"Unexpected feature dim {C} in {split}/{tk}")

            # channel mean/var accumulation
            x_np = x[()]  # per-trial load only
            sum_x += x_np.sum(axis=0, dtype=np.float64)
            sum_x2 += (x_np * x_np).sum(axis=0, dtype=np.float64)
            n_total += int(T)

            # correlation sampling: choose up to corr_timepoints_per_trial rows
            if T > 0:
                k = min(corr_timepoints_per_trial, T)
                idx = rng.choice(T, size=k, replace=False) if T >= k else np.arange(T)
                corr_rows.append(x_np[idx, :].astype(np.float32))

            # temporal energy
            e = np.sqrt((x_np * x_np).sum(axis=1, dtype=np.float64))
            _accum_energy(e)

    mean = sum_x / max(n_total, 1)
    var = (sum_x2 / max(n_total, 1)) - (mean * mean)
    var = np.maximum(var, 0.0)

    df_ch = pd.DataFrame(
        {
            "channel": np.arange(512, dtype=np.int32),
            "mean": mean.astype(np.float64),
            "var": var.astype(np.float64),
        }
    )
    ch_csv = outdir / "channel_mean_var.csv"
    df_ch.to_csv(ch_csv, index=False)
    print(f"Wrote: {ch_csv}")

    # Plot variance across channels
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df_ch["channel"].values, df_ch["var"].values, linewidth=1.0)
    ax1.set_title("Per-channel variance across sampled trials/timepoints")
    ax1.set_xlabel("channel")
    ax1.set_ylabel("variance")
    _save_fig(fig1, outdir, "channel_variance")
    plt.close(fig1)

    # Correlation matrix heatmap
    corr_mat = None
    if len(corr_rows) > 0:
        X = np.vstack(corr_rows)  # (N, 512), N limited by 200*25=5000
        # Centering improves numeric stability
        X = X - X.mean(axis=0, keepdims=True)
        corr_mat = np.corrcoef(X, rowvar=False)

        fig2, ax2 = plt.subplots(figsize=(8, 7))
        im = ax2.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        ax2.set_title("Channel correlation heatmap (subsampled timepoints)")
        ax2.set_xlabel("channel")
        ax2.set_ylabel("channel")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        _save_fig(fig2, outdir, "channel_correlation_heatmap")
        plt.close(fig2)

    # Temporal energy mean/std curve
    valid = energy_count > 0
    t_idx = np.arange(energy_sum.shape[0], dtype=np.int32)
    mean_e = np.zeros_like(energy_sum, dtype=np.float64)
    std_e = np.zeros_like(energy_sum, dtype=np.float64)
    mean_e[valid] = energy_sum[valid] / energy_count[valid]
    var_e = np.zeros_like(energy_sum, dtype=np.float64)
    var_e[valid] = (energy_sum2[valid] / energy_count[valid]) - (mean_e[valid] * mean_e[valid])
    var_e = np.maximum(var_e, 0.0)
    std_e[valid] = np.sqrt(var_e[valid])

    df_energy = pd.DataFrame(
        {
            "t": t_idx,
            "count": energy_count.astype(np.int64),
            "mean_energy": mean_e.astype(np.float64),
            "std_energy": std_e.astype(np.float64),
        }
    )
    energy_csv = outdir / "temporal_energy.csv"
    df_energy.to_csv(energy_csv, index=False)
    print(f"Wrote: {energy_csv}")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(t_idx[valid], mean_e[valid], color="black", linewidth=1.2, label="mean")
    ax3.fill_between(
        t_idx[valid],
        (mean_e[valid] - std_e[valid]),
        (mean_e[valid] + std_e[valid]),
        color="gray",
        alpha=0.3,
        label="±1 std",
    )
    ax3.set_title("Temporal energy (mean L2 norm across channels vs time)")
    ax3.set_xlabel("time index")
    ax3.set_ylabel("L2 norm")
    ax3.legend(loc="best")
    _save_fig(fig3, outdir, "temporal_energy_curve")
    plt.close(fig3)

    print(f"Wrote figures to: {outdir}")


if __name__ == "__main__":
    main()

