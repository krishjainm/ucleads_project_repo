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


def _label_len(seq_class_ids: np.ndarray):
    # Assumption: 0 indicates padding (common for fixed-length targets).
    return int(np.sum(seq_class_ids != 0))


def _transcript_len(transcription: np.ndarray):
    # Assumption: transcription is ASCII codes padded with 0.
    return int(np.sum(transcription != 0))


def collect_lengths(h5_path: Path, split_name: str):
    rows = []
    with h5py.File(h5_path, "r") as f:
        trials = _trial_keys(f)
        if len(trials) == 0:
            raise RuntimeError(f"No trial_* groups found in {h5_path}")

        for tk in trials:
            g = f[tk]
            x = g["input_features"]
            y = g["seq_class_ids"][()]
            t = g["transcription"][()]

            rows.append(
                {
                    "split": split_name,
                    "trial_key": tk,
                    "neural_len": int(x.shape[0]),
                    "label_len": _label_len(y),
                    "transcript_len": _transcript_len(t),
                }
            )

    return rows


def _save_fig(fig, outdir: Path, stem: str):
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description="Phase 1: plot trial length structure")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()

    train_path, val_path = _hdf5_paths(args.data_root, args.date)
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val file: {val_path}")

    outdir = _ensure_outdir()

    rows = []
    rows += collect_lengths(train_path, "train")
    rows += collect_lengths(val_path, "val")

    df = pd.DataFrame(rows)
    csv_path = outdir / "trial_lengths.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")

    # 1) Histogram neural_len
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.hist(df["neural_len"].values, bins=50, color="steelblue", alpha=0.85)
    ax1.set_title("Histogram: neural_len (input_features T)")
    ax1.set_xlabel("neural_len")
    ax1.set_ylabel("count")
    _save_fig(fig1, outdir, "hist_neural_len")
    plt.close(fig1)

    # 2) Histogram label_len
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(df["label_len"].values, bins=50, color="darkorange", alpha=0.85)
    ax2.set_title("Histogram: label_len (non-pad seq_class_ids)")
    ax2.set_xlabel("label_len")
    ax2.set_ylabel("count")
    _save_fig(fig2, outdir, "hist_label_len")
    plt.close(fig2)

    # 3) Scatter neural_len vs label_len
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.scatter(df["neural_len"].values, df["label_len"].values, s=8, alpha=0.35)
    ax3.set_title("Scatter: neural_len vs label_len")
    ax3.set_xlabel("neural_len")
    ax3.set_ylabel("label_len")
    _save_fig(fig3, outdir, "scatter_neural_len_vs_label_len")
    plt.close(fig3)

    print(f"Wrote figures to: {outdir}")


if __name__ == "__main__":
    main()

