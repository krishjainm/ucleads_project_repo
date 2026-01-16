import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------

def trim_zeros(arr: np.ndarray) -> np.ndarray:
    """Trim trailing zeros from a 1D int array. If all zeros, return empty."""
    arr = np.asarray(arr).reshape(-1)
    nz = np.nonzero(arr)[0]
    if len(nz) == 0:
        return arr[:0].astype(int)
    return arr[: nz[-1] + 1].astype(int)

def load_gt(hdf5_path: str):
    """Return dict trial_id -> trimmed transcription (1D int)."""
    gt = {}
    with h5py.File(hdf5_path, "r") as f:
        for k in f.keys():
            # trial keys are like trial_0001 etc
            try:
                y = f[k]["transcription"][()]
            except Exception:
                continue
            gt[k] = trim_zeros(y)
    return gt

def load_preds(pred_path: str):
    """
    predictions.npy is expected to be an object array/list of (trial_id, pred_array)
    where pred_array may be shape (N,) or (1,N) or list-like.
    """
    p = np.load(pred_path, allow_pickle=True)
    preds = {}
    for tid, arr in p:
        preds[str(tid)] = np.asarray(arr).reshape(-1).astype(int)
    return preds

def levenshtein_ops(a: np.ndarray, b: np.ndarray):
    """
    Compute Levenshtein distance plus backtrace counts of insertions, deletions, substitutions.
    a = pred, b = gt
    Returns: (dist, ins, del, sub)
    """
    a = a.tolist()
    b = b.tolist()
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # deletion
                dp[i, j - 1] + 1,      # insertion
                dp[i - 1, j - 1] + cost  # substitution/match
            )

    # backtrace for op counts
    i, j = n, m
    ins = dele = sub = 0
    while i > 0 or j > 0:
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            dele += 1
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            ins += 1
            j -= 1
        else:
            # diag
            if i > 0 and j > 0 and a[i - 1] != b[j - 1]:
                sub += 1
            i -= 1
            j -= 1

    return int(dp[n, m]), int(ins), int(dele), int(sub)

def summarize_sequence(seq: np.ndarray, max_len=60):
    """Short textual summary: first tokens + length + unique count."""
    seq = np.asarray(seq).reshape(-1).astype(int)
    head = seq[:max_len].tolist()
    return {
        "len": int(len(seq)),
        "unique": int(len(set(head))) if len(head) > 0 else 0,
        "head": " ".join(map(str, head))
    }

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="Path to predictions.npy")
    ap.add_argument("--val-hdf5", required=True, help="Path to data_val.hdf5")
    ap.add_argument("--out-dir", default="figures/phase3", help="Output directory")
    ap.add_argument("--top-k", type=int, default=10, help="How many best/worst to print")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt(args.val_hdf5)
    preds = load_preds(args.predictions)

    rows = []
    missing = 0
    for tid, y in gt.items():
        if tid not in preds:
            missing += 1
            continue
        p = preds[tid]
        dist, ins, dele, sub = levenshtein_ops(p, y)
        gt_len = max(1, len(y))
        ter = dist / gt_len

        # extra failure-mode descriptors
        pred_zero_frac = float(np.mean(p == 0)) if len(p) else 0.0
        pred_mode = int(pd.Series(p).mode().iloc[0]) if len(p) else -1

        rows.append({
            "trial_id": tid,
            "gt_len": len(y),
            "pred_len": len(p),
            "edit_dist": dist,
            "TER": ter,
            "ins": ins,
            "del": dele,
            "sub": sub,
            "pred_zero_frac": pred_zero_frac,
            "pred_mode_token": pred_mode
        })

    df = pd.DataFrame(rows).sort_values("TER", ascending=True)
    df.to_csv(out_dir / "phase3_trial_error_table.csv", index=False)

    # Print qualitative examples (best and worst)
    best = df.head(args.top_k)
    worst = df.tail(args.top_k).sort_values("TER", ascending=False)

    report_lines = []
    report_lines.append(f"Missing predictions for {missing} val trials.\n")
    report_lines.append("=== BEST TRIALS (lowest TER) ===\n")
    for _, r in best.iterrows():
        tid = r["trial_id"]
        s_gt = summarize_sequence(gt[tid])
        s_pr = summarize_sequence(preds[tid])
        report_lines.append(
            f"{tid} TER={r['TER']:.3f} edits={r['edit_dist']} (ins={r['ins']}, del={r['del']}, sub={r['sub']})\n"
            f"  GT  len={s_gt['len']} head: {s_gt['head']}\n"
            f"  PR  len={s_pr['len']} head: {s_pr['head']}\n"
        )

    report_lines.append("\n=== WORST TRIALS (highest TER) ===\n")
    for _, r in worst.iterrows():
        tid = r["trial_id"]
        s_gt = summarize_sequence(gt[tid])
        s_pr = summarize_sequence(preds[tid])
        report_lines.append(
            f"{tid} TER={r['TER']:.3f} edits={r['edit_dist']} (ins={r['ins']}, del={r['del']}, sub={r['sub']})\n"
            f"  GT  len={s_gt['len']} head: {s_gt['head']}\n"
            f"  PR  len={s_pr['len']} head: {s_pr['head']}\n"
        )

    (out_dir / "phase3_qualitative_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    # Plot 1: TER vs gt_len
    plt.figure()
    plt.scatter(df["gt_len"], df["TER"], s=10)
    plt.xlabel("Ground truth length (trimmed)")
    plt.ylabel("TER")
    plt.title("Phase 3: TER vs GT length")
    plt.tight_layout()
    plt.savefig(out_dir / "ter_vs_gt_len_phase3.png", dpi=200)
    plt.close()

    # Plot 2: TER vs pred_len
    plt.figure()
    plt.scatter(df["pred_len"], df["TER"], s=10)
    plt.xlabel("Predicted length")
    plt.ylabel("TER")
    plt.title("Phase 3: TER vs predicted length")
    plt.tight_layout()
    plt.savefig(out_dir / "ter_vs_pred_len_phase3.png", dpi=200)
    plt.close()

    # Plot 3: Edit operation composition (ins/del/sub fractions)
    denom = (df["ins"] + df["del"] + df["sub"]).replace(0, 1)
    comp = pd.DataFrame({
        "ins_frac": df["ins"] / denom,
        "del_frac": df["del"] / denom,
        "sub_frac": df["sub"] / denom
    })
    plt.figure()
    plt.hist(comp["ins_frac"], bins=30, alpha=0.7, label="ins")
    plt.hist(comp["del_frac"], bins=30, alpha=0.7, label="del")
    plt.hist(comp["sub_frac"], bins=30, alpha=0.7, label="sub")
    plt.xlabel("Fraction of edits per trial")
    plt.ylabel("Count")
    plt.title("Phase 3: Edit operation composition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "edit_op_fractions_hist.png", dpi=200)
    plt.close()

    # Plot 4: Pred dominant-token indicator
    plt.figure()
    plt.hist(df["pred_mode_token"], bins=50)
    plt.xlabel("Mode token in prediction")
    plt.ylabel("Count")
    plt.title("Phase 3: Prediction mode-token distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_mode_token_hist.png", dpi=200)
    plt.close()

    print(f"[OK] Wrote: {out_dir / 'phase3_trial_error_table.csv'}")
    print(f"[OK] Wrote: {out_dir / 'phase3_qualitative_report.txt'}")
    print(f"[OK] Wrote Phase 3 plots to: {out_dir}")

if __name__ == "__main__":
    main()
