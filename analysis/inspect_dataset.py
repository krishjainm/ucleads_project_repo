import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _hdf5_paths(data_root: str, date: str):
    base = Path(data_root) / date
    return base / "data_train.hdf5", base / "data_val.hdf5"


def _trial_keys(f: h5py.File):
    return sorted([k for k in f.keys() if k.startswith("trial_")])


def _ensure_outdir():
    outdir = Path("figures") / "phase1"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def inspect_file(h5_path: Path, split_name: str):
    with h5py.File(h5_path, "r") as f:
        trials = _trial_keys(f)
        if len(trials) == 0:
            raise RuntimeError(f"No trial_* groups found in {h5_path}")

        example_keys = trials[:5]

        # Inspect 3 trials for shapes/dtypes + feature-dim validation
        inspected = []
        feature_dim_bad = 0
        feature_dim_examples = []

        for i, tk in enumerate(trials):
            g = f[tk]
            x = g["input_features"]
            y = g["seq_class_ids"]
            t = g["transcription"]

            # feature dim check
            if x.shape[-1] != 512:
                feature_dim_bad += 1
                if len(feature_dim_examples) < 5:
                    feature_dim_examples.append((tk, tuple(x.shape)))

            if len(inspected) < 3:
                inspected.append(
                    {
                        "split": split_name,
                        "trial_key": tk,
                        "input_features_shape": tuple(x.shape),
                        "input_features_dtype": str(x.dtype),
                        "seq_class_ids_shape": tuple(y.shape),
                        "seq_class_ids_dtype": str(y.dtype),
                        "transcription_shape": tuple(t.shape),
                        "transcription_dtype": str(t.dtype),
                    }
                )

        summary = {
            "split": split_name,
            "hdf5_path": str(h5_path),
            "num_trials": len(trials),
            "example_trial_keys": "|".join(example_keys),
            "feature_dim_expected": 512,
            "feature_dim_violations": feature_dim_bad,
            "feature_dim_violation_examples": "|".join(
                [f"{k}:{s}" for k, s in feature_dim_examples]
            ),
        }

    return summary, inspected


def main():
    parser = argparse.ArgumentParser(description="Phase 1: inspect HDF5 dataset structure")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()

    train_path, val_path = _hdf5_paths(args.data_root, args.date)
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val file: {val_path}")

    outdir = _ensure_outdir()

    train_summary, train_inspected = inspect_file(train_path, "train")
    val_summary, val_inspected = inspect_file(val_path, "val")

    # Console output
    print("=== Dataset counts ===")
    print(f"Train trials: {train_summary['num_trials']}")
    print(f"Val trials:   {val_summary['num_trials']}")
    print()

    print("=== Example trial keys ===")
    print(f"Train: {train_summary['example_trial_keys']}")
    print(f"Val:   {val_summary['example_trial_keys']}")
    print()

    print("=== Shapes/dtypes for 3 trials (per split) ===")
    for row in (train_inspected + val_inspected):
        print(f"[{row['split']}] {row['trial_key']}")
        print(f"  input_features: {row['input_features_shape']} {row['input_features_dtype']}")
        print(f"  seq_class_ids:  {row['seq_class_ids_shape']} {row['seq_class_ids_dtype']}")
        print(f"  transcription:  {row['transcription_shape']} {row['transcription_dtype']}")
    print()

    print("=== Feature-dim (expected 512) ===")
    print(
        f"Train violations: {train_summary['feature_dim_violations']} "
        f"(examples: {train_summary['feature_dim_violation_examples'] or 'none'})"
    )
    print(
        f"Val violations:   {val_summary['feature_dim_violations']} "
        f"(examples: {val_summary['feature_dim_violation_examples'] or 'none'})"
    )

    # CSV summary
    df_summary = pd.DataFrame([train_summary, val_summary])
    csv_path = outdir / "dataset_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print()
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()

