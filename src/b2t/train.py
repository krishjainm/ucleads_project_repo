import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Dataset
# -------------------------
class BrainTextDataset(Dataset):
    """
    HDF5 structure per trial:
      trial_0000/
        input_features (T, 512) float32
        transcription   (L,) int32   (preferred)
        seq_class_ids   (L,) int32   (fallback)
    """

    def __init__(self, hdf5_path, norm_stats_path=None, target_key="transcription"):
        self.hdf5_path = str(hdf5_path)
        self.target_key = target_key

        self.mean = None
        self.std = None
        if norm_stats_path is not None:
            d = np.load(norm_stats_path)
            self.mean = d["mean"].astype(np.float32)
            self.std = d["std"].astype(np.float32)

        with h5py.File(self.hdf5_path, "r") as f:
            self.trials = sorted([k for k in f.keys() if k.startswith("trial_")])

        if len(self.trials) == 0:
            raise RuntimeError(f"No trial_* groups found in {self.hdf5_path}")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        tid = self.trials[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            g = f[tid]
            x = g["input_features"][()].astype(np.float32)  # (T,512)

            # Prefer transcription, fallback to seq_class_ids
            if self.target_key in g:
                y = g[self.target_key][()].astype(np.int64)
            else:
                y = g["seq_class_ids"][()].astype(np.int64)

        # Many datasets use 0 as padding, remove all zeros.
        if y.size > 0:
            y = y[y != 0]

        # Remove any blank tokens from targets, CTC blank is 0 and must not appear in targets
        if y.size > 0:
            y = y[y != 0]

        # Normalize x if stats provided
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)

        return torch.from_numpy(x).float(), torch.from_numpy(y).long()

# -------------------------
# Model (BiGRU + LayerNorm)
# -------------------------
class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=41, num_layers=2, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.ln = nn.LayerNorm(out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)      # (B,T,H or 2H)
        out = self.ln(out)        # (B,T,feat)
        logits = self.fc(out)     # (B,T,C)
        return logits


# -------------------------
# Collate for CTC
# -------------------------
def collate_fn(batch):
    """
    Pads inputs to max T in batch.
    For CTC: concatenates targets into 1D vector and returns lengths.
    """
    xs, ys = zip(*batch)

    lengths_x = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    lengths_y = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)

    max_T = int(lengths_x.max().item())
    feat_dim = xs[0].shape[1]

    x_padded = torch.zeros(len(xs), max_T, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        x_padded[i, : x.shape[0], :] = x

    y_concat = torch.cat(ys, dim=0).to(torch.long)

    return x_padded, y_concat, lengths_x, lengths_y

# -------------------------
# Optional train augmentation: temporal masking
# -------------------------
def apply_time_mask(x, x_lens, p=0.5, count=2, max_width=20):
    """
    x: (B,T,D)
    x_lens: (B,)
    With prob p, mask 1..count spans of width up to max_width per sequence.
    """
    if p <= 0:
        return x
    B, T, D = x.shape
    for b in range(B):
        if np.random.rand() > p:
            continue
        Tb = int(x_lens[b].item())
        if Tb <= 1:
            continue
        nspans = np.random.randint(1, count + 1)
        for _ in range(nspans):
            w = np.random.randint(1, max_width + 1)
            if Tb - w <= 0:
                continue
            s = np.random.randint(0, Tb - w)
            x[b, s : s + w, :] = 0.0
    return x


# -------------------------
# Train epoch (CTC)
# -------------------------
def train_epoch(
    model, dataloader, criterion, optimizer, device,
    grad_clip=1.0,
    time_mask_p=0.0,
    time_mask_count=2,
    time_mask_max_width=20,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for neural, labels, neural_lengths, label_lengths in dataloader:
        neural = neural.to(device)
        labels = labels.to(device)
        neural_lengths = neural_lengths.to(device)
        label_lengths = label_lengths.to(device)

        # Apply temporal masking (train only)
        if time_mask_p > 0:
            neural = apply_time_mask(
                neural, neural_lengths,
                p=time_mask_p,
                count=time_mask_count,
                max_width=time_mask_max_width
            )

        # Drop invalid samples for CTC, need 0 < U <= T
        keep = (label_lengths > 0) & (label_lengths <= neural_lengths)
        if keep.sum().item() == 0:
            continue

        neural = neural[keep]
        neural_lengths_kept = neural_lengths[keep]
        label_lengths_kept = label_lengths[keep]

        # Rebuild concatenated labels for kept samples
        kept = []
        idx = 0
        for b in range(len(label_lengths)):
            L = int(label_lengths[b].item())
            if keep[b].item():
                kept.append(labels[idx:idx + L])
            idx += L

        labels_kept = torch.cat(kept, dim=0) if len(kept) else torch.zeros((0,), dtype=torch.long, device=device)

        optimizer.zero_grad()

        logits = model(neural)                  # (B,T,C)
        logits = logits.transpose(0, 1)         # (T,B,C)
        log_probs = torch.log_softmax(logits, dim=2)

        loss = criterion(log_probs, labels_kept, neural_lengths_kept, label_lengths_kept)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")

# -------------------------
# Norm stats
# -------------------------
def compute_and_save_norm_stats(train_h5_path: str, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_sum = np.zeros(512, dtype=np.float64)
    total_sumsq = np.zeros(512, dtype=np.float64)
    total_count = 0

    with h5py.File(train_h5_path, "r") as f:
        trial_ids = [k for k in f.keys() if k.startswith("trial_")]
        for tid in trial_ids:
            x = f[tid]["input_features"][()].astype(np.float64)
            total_sum += x.sum(axis=0)
            total_sumsq += (x * x).sum(axis=0)
            total_count += x.shape[0]

    mean = total_sum / max(1, total_count)
    var = total_sumsq / max(1, total_count) - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    np.savez_compressed(out_path, mean=mean, std=std)
    print("[OK] wrote norm stats to", str(out_path))
    print("mean shape", mean.shape, "std shape", std.shape, "std min", float(std.min()), "std max", float(std.max()))


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train brain-to-text model (CTC)")

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")

    # normalization
    parser.add_argument("--compute-norm-stats", action="store_true")
    parser.add_argument("--norm-stats", type=str, default="outputs/phase5/norm_stats.npz")

    # stabilization
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # scheduler
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-factor", type=float, default=0.5)

    # temporal mask augmentation (train only)
    parser.add_argument("--time-mask-p", type=float, default=0.0)
    parser.add_argument("--time-mask-count", type=int, default=2)
    parser.add_argument("--time-mask-max-width", type=int, default=20)

    # target source
    parser.add_argument("--target-key", type=str, default="seq_class_ids",
                    help="HDF5 key to use as target, usually 'transcription' or 'seq_class_ids'.")

    args = parser.parse_args()

    train_path = os.path.join(args.data_root, args.date, "data_train.hdf5")
    val_path = os.path.join(args.data_root, args.date, "data_val.hdf5")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    if args.compute_norm_stats:
        compute_and_save_norm_stats(train_path, args.norm_stats)
        return

    norm_path = args.norm_stats if (args.norm_stats and os.path.exists(args.norm_stats)) else None

    train_ds = BrainTextDataset(train_path, norm_stats_path=norm_path, target_key=args.target_key)
    val_ds = BrainTextDataset(val_path, norm_stats_path=norm_path, target_key=args.target_key)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    if args.dry_run:
        x, y, xl, yl = next(iter(train_loader))
        print("DRY RUN OK")
        print("x:", x.shape, x.dtype, "y:", y.shape, y.dtype)
        print("x_lens:", xl[:5].tolist(), "y_lens:", yl[:5].tolist())
        return

    device = torch.device(args.device)
    model = SimpleRNNModel(
        input_dim=512,
        hidden_dim=args.hidden_dim,
        num_classes=41,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
    ).to(device)

    print("Model params:", sum(p.numel() for p in model.parameters()))

    # Proper CTC again
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.plateau_factor, patience=args.plateau_patience
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=args.grad_clip,
            time_mask_p=args.time_mask_p,
            time_mask_count=args.time_mask_count,
            time_mask_max_width=args.time_mask_max_width,
        )
        scheduler.step(train_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - LR: {lr_now:.6f}")

    ckpt_path = os.path.join(args.checkpoint_dir, "final_checkpoint.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    print("Checkpoint saved to", ckpt_path)


if __name__ == "__main__":
    main()
