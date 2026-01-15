import argparse
import csv
import numpy as np
import h5py


def trim_zeros_right(arr):
    arr = np.asarray(arr).reshape(-1).astype(int)
    # keep everything up to last nonzero
    nz = np.nonzero(arr)[0]
    if len(nz) == 0:
        return []
    return arr[: nz[-1] + 1].tolist()


def edit_distance(a, b):
    # Levenshtein distance
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # delete
                dp[j - 1] + 1,  # insert
                prev + cost     # substitute
            )
            prev = cur
    return dp[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-npy", required=True)
    ap.add_argument("--val-hdf5", required=True)
    ap.add_argument("--csv-out", required=True)
    args = ap.parse_args()

    preds = np.load(args.pred_npy, allow_pickle=True)
    pred_map = {tid: seq for tid, seq in preds}

    rows = []
    total_edits = 0
    total_len = 0
    missing = 0

    with h5py.File(args.val_hdf5, "r") as f:
        trial_ids = list(f.keys())
        for tid in trial_ids:
            gt = trim_zeros_right(f[tid]["transcription"][:])
            if tid not in pred_map:
                missing += 1
                continue
            pred = list(map(int, pred_map[tid]))

            ed = edit_distance(gt, pred)
            denom = max(1, len(gt))
            ter = ed / denom

            rows.append([tid, len(gt), len(pred), ed, ter])
            total_edits += ed
            total_len += len(gt)

    overall_ter = total_edits / max(1, total_len)
    with open(args.csv_out, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["trial_id", "gt_len", "pred_len", "edit_distance", "ter"])
        w.writerows(rows)

    print(f"Wrote: {args.csv_out}")
    print(f"VAL TER: {overall_ter:.4f} (lower is better)")
    print(f"Scored {len(rows)} trials, missing preds for {missing}")


if __name__ == "__main__":
    main()
