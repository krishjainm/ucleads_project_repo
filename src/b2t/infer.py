import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import h5py


class SimpleRNNModel(nn.Module):
    """GRU encoder + linear classifier (same idea as train.py)."""

    def __init__(self, input_dim=512, hidden_dim=256, num_classes=41, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)          # (B, T, H)
        logits = self.fc(out)         # (B, T, C)
        return logits


def load_feature_dim_any_trial(hdf5_path: str) -> int:
    """Peek one trial to infer feature dimension (should be 512)."""
    with h5py.File(hdf5_path, "r") as f:
        trial_keys = list(f.keys())
        if len(trial_keys) == 0:
            raise RuntimeError(f"No trials found in {hdf5_path}")
        t0 = trial_keys[0]
        x = f[t0]["input_features"]
        # expected (T, D)
        return int(x.shape[-1])


def greedy_decode(logits: torch.Tensor, blank_id: int = 0) -> np.ndarray:
    """
    logits: (T, C) torch tensor
    returns: 1D numpy array of predicted token ids after basic CTC-style collapse:
      - argmax per frame
      - remove consecutive duplicates
      - remove blanks
    """
    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(int).tolist()

    collapsed = []
    prev = None
    for p in pred:
        if prev is None or p != prev:
            collapsed.append(p)
        prev = p

    collapsed = [p for p in collapsed if p != blank_id]
    return np.array(collapsed, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Run inference on brain-to-text validation split")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing HDF5 data (folder that contains date folders)")
    parser.add_argument("--date", type=str, required=True,
                        help="Date folder name (e.g., t15.2023.10.06)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for predictions.npy")

    # Must match training for each ablation
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="GRU hidden dimension (must match checkpoint)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of GRU layers (must match checkpoint)")

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cuda or cpu")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "train", "test"],
                        help="Which split to run inference on (default val)")

    args = parser.parse_args()

    # Pick split file
    split_name = {"train": "data_train.hdf5", "val": "data_val.hdf5", "test": "data_test.hdf5"}[args.split]
    hdf5_path = os.path.join(args.data_root, args.date, split_name)
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Split file not found: {hdf5_path}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Infer hyperparams if checkpoint stored args
    ckpt_args = None
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        ckpt_args = ckpt["args"]

    hidden_dim = args.hidden_dim
    num_layers = args.num_layers

    if hidden_dim is None and ckpt_args is not None and "hidden_dim" in ckpt_args:
        hidden_dim = int(ckpt_args["hidden_dim"])
    if num_layers is None and ckpt_args is not None and "num_layers" in ckpt_args:
        num_layers = int(ckpt_args["num_layers"])

    if hidden_dim is None or num_layers is None:
        raise ValueError(
            "You must provide --hidden-dim and --num-layers, or the checkpoint must contain them in ckpt['args']."
        )

    # Build model with correct shapes
    input_dim = load_feature_dim_any_trial(hdf5_path)
    num_classes = 41
    device = torch.device(args.device)

    model = SimpleRNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
    ).to(device)

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print("Warning: state_dict mismatch")
        print("  Missing keys:", missing)
        print("  Unexpected keys:", unexpected)

    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "predictions.npy")

    preds = []
    with h5py.File(hdf5_path, "r") as f:
        trial_ids = list(f.keys())

        for tid in trial_ids:
            x = f[tid]["input_features"][:]  # (T, 512)
            x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)  # (1, T, 512)

            with torch.no_grad():
                logits = model(x_t).squeeze(0)  # (T, C)

            pred_ids = greedy_decode(logits, blank_id=0)
            preds.append((tid, pred_ids))

    np.save(out_path, np.array(preds, dtype=object), allow_pickle=True)
    print(f"Saved predictions to {out_path}")
    print(f"Trials: {len(preds)}")
    if len(preds) > 0:
        print("Example:", preds[0][0], "len", int(preds[0][1].shape[0]))


if __name__ == "__main__":
    main()

