import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-dir", required=True)
    ap.add_argument("--out-dir", default="figures/phase4")
    ap.add_argument("--max-points", type=int, default=20000)
    args = ap.parse_args()

    hidden_dir = Path(args.hidden_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(hidden_dir.glob("*.npz"))
    assert len(files) > 0, "No hidden state files found"

    all_h = []
    all_t = []
    all_trial = []

    for ti, fp in enumerate(files):
        d = np.load(fp)
        h = d["hidden"]
        T = h.shape[0]

        all_h.append(h)
        all_t.append(np.arange(T))
        all_trial.append(np.full(T, ti))

    H = np.vstack(all_h)
    t = np.concatenate(all_t)
    trial = np.concatenate(all_trial)

    if len(H) > args.max_points:
        idx = np.random.choice(len(H), args.max_points, replace=False)
        H = H[idx]
        t = t[idx]
        trial = trial[idx]

    pca = PCA(n_components=2)
    Z = pca.fit_transform(H)

    # Plot 1: color by time
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=t, s=5)
    plt.colorbar(sc, label="Time index")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Hidden state PCA (colored by time)")
    plt.tight_layout()
    plt.savefig(out_dir / "hidden_pca_time.png", dpi=200)
    plt.close()

    # Plot 2: color by trial
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=trial, s=5)
    plt.colorbar(sc, label="Trial index")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Hidden state PCA (colored by trial)")
    plt.tight_layout()
    plt.savefig(out_dir / "hidden_pca_trial.png", dpi=200)
    plt.close()

    ve = pca.explained_variance_ratio_
    (out_dir / "pca_variance_explained.txt").write_text(
        f"PC1: {ve[0]:.4f}\nPC2: {ve[1]:.4f}\nPC1+PC2: {ve[0]+ve[1]:.4f}\n"
    )

    print("[OK] PCA plots written to", out_dir)

if __name__ == "__main__":
    main()
