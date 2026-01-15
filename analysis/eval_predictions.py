import argparse
import os
import csv
import numpy as np

def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-npy", required=True, help="Path to predictions.npy")
    ap.add_argument("--csv-out", required=True, help="Where to write metrics CSV")
    args = ap.parse_args()

    preds = np.load(args.pred_npy, allow_pickle=True)

    # predictions.npy format: list of (trial_id, pred_seq)
    # we do not have GT here yet, so we compute:
    # 1) length stats
    # 2) duplicate rate
    # 3) pairwise similarity sample (proxy)
    lengths = []
    uniq = set()
    dup = 0

    for trial_id, arr in preds:
        uniq.add(trial_id)
        seq = np.asarray(arr).reshape(-1).astype(int).tolist()
        lengths.append(len(seq))

    # quick proxy: compare consecutive sequences edit distance (sample first 30)
    sample = []
    for i in range(min(30, len(preds) - 1)):
        a = np.asarray(preds[i][1]).reshape(-1).astype(int).tolist()
        b = np.asarray(preds[i + 1][1]).reshape(-1).astype(int).tolist()
        d = levenshtein(a, b)
        sample.append(d)

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_trials", "min_len", "max_len", "mean_len", "median_len", "sample_mean_edit_dist"])
        w.writerow([
            len(preds),
            int(np.min(lengths)),
            int(np.max(lengths)),
            float(np.mean(lengths)),
            float(np.median(lengths)),
            float(np.mean(sample)) if sample else ""
        ])

    print(f"Wrote: {args.csv_out}")

if __name__ == "__main__":
    main()
