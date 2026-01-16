import argparse
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn as nn

def trim_zeros(arr):
    arr = np.asarray(arr).reshape(-1)
    nz = np.nonzero(arr)[0]
    if len(nz) == 0:
        return arr[:0].astype(int)
    return arr[: nz[-1] + 1].astype(int)

class GRUModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1, vocab_size=512):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        h_seq, _ = self.gru(x)   # (B, T, H)
        logits = self.fc(h_seq)
        return logits, h_seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-hdf5", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--num-trials", type=int, default=10)
    ap.add_argument("--out-dir", default="outputs/phase4/hidden_states")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUModel(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    with h5py.File(args.val_hdf5, "r") as f:
        trial_ids = list(f.keys())[: args.num_trials]
        for tid in trial_ids:
            x = f[tid]["input_features"][()].astype(np.float32)
            y = trim_zeros(f[tid]["transcription"][()])

            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            with torch.no_grad():
                _, h_seq = model(xt)

            h = h_seq.squeeze(0).cpu().numpy()

            np.savez_compressed(
                out_dir / f"{tid}.npz",
                hidden=h,
                gt=y,
                T=h.shape[0],
                H=h.shape[1]
            )

    print(f"[OK] Saved hidden states for {len(trial_ids)} trials")

if __name__ == "__main__":
    main()
