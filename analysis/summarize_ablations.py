import pandas as pd
from pathlib import Path

configs = [
    ("GRU-1L-256", "outputs/ablations/gru1_hd256_bs8_e10/val_scoring.csv"),
    ("GRU-2L-256", "outputs/ablations/gru2_hd256_bs8_e10/val_scoring.csv"),
    ("GRU-2L-512", "outputs/ablations/gru2_hd512_bs8_e10/val_scoring.csv"),
]

rows = []

for name, path in configs:
    df = pd.read_csv(path)
    rows.append({
        "model": name,
        "mean_TER": df.ter.mean(),
        "median_TER": df.ter.median(),
        "min_TER": df.ter.min(),
        "max_TER": df.ter.max(),
    })

out = pd.DataFrame(rows)
out.to_csv("figures/phase2/ablation_summary.csv", index=False)
print(out)
