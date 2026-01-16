import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn


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
            bidirectional=bidirectional
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.ln = nn.LayerNorm(out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.ln(out)
        logits = self.fc(out)
        return logits


def ctc_greedy_decode(log_probs, blank=0):
    """
    log_probs: (T, C) numpy array of log-probabilities
    returns: list[int]
    """
    best = np.argmax(log_probs, axis=1).tolist()
    out = []
    prev = None
    for b in best:
        if b == blank:
            prev = b
            continue
        if prev is None or b != prev:
            out.append(int(b))
        prev = b
    return out


def _logsumexp(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = a if a > b else b
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def ctc_prefix_beam_search(log_probs, beam_width=20, blank=0):
    """
    Prefix beam search for CTC without external deps.
    log_probs: (T, C) numpy array of log-probabilities
    Returns: list[int] best token sequence (no blanks)
    """
    T, C = log_probs.shape

    # beams maps prefix -> (p_blank, p_nonblank)
    beams = {(): (0.0, -np.inf)}

    for t in range(T):
        next_beams = {}

        for prefix, (p_b, p_nb) in beams.items():
            for c in range(C):
                p = float(log_probs[t, c])

                if c == blank:
                    # emit blank, prefix unchanged
                    pb2, pnb2 = next_beams.get(prefix, (-np.inf, -np.inf))
                    pb2 = _logsumexp(pb2, p_b + p)
                    pb2 = _logsumexp(pb2, p_nb + p)
                    next_beams[prefix] = (pb2, pnb2)
                    continue

                last = prefix[-1] if len(prefix) > 0 else None

                if c == last:
                    # extend with same char: only from blank keeps prefix unchanged
                    pb2, pnb2 = next_beams.get(prefix, (-np.inf, -np.inf))
                    pnb2 = _logsumexp(pnb2, p_b + p)
                    next_beams[prefix] = (pb2, pnb2)

                    # also allow staying in same prefix from nonblank (does not add symbol)
                    pb3, pnb3 = next_beams.get(prefix, (-np.inf, -np.inf))
                    pnb3 = _logsumexp(pnb3, p_nb + p)
                    next_beams[prefix] = (pb3, pnb3)
                else:
                    new_prefix = prefix + (c,)
                    pb2, pnb2 = next_beams.get(new_prefix, (-np.inf, -np.inf))
                    p_total = _logsumexp(p_b, p_nb)
                    pnb2 = _logsumexp(pnb2, p_total + p)
                    next_beams[new_prefix] = (pb2, pnb2)

        # prune to top beams by total score
        def total_score(item):
            pb, pnb = item[1]
            return _logsumexp(pb, pnb)

        beams = dict(sorted(next_beams.items(), key=total_score, reverse=True)[:beam_width])

    best_prefix = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
    return [int(x) for x in best_prefix]

def main():
    parser = argparse.ArgumentParser(description="Infer brain-to-text model")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Phase 5.2.1
    parser.add_argument("--norm-stats", type=str, default="outputs/phase5/norm_stats.npz")
        # Phase 5.2.3
    parser.add_argument("--beam-width", type=int, default=1, help="CTC beam width. 1 = greedy")

    args = parser.parse_args()

    val_h5 = Path(args.data_root) / args.date / "data_val.hdf5"
    if not val_h5.exists():
        raise FileNotFoundError(f"Validation HDF5 not found: {val_h5}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load normalization stats (mean/std over train)
    norm = np.load(args.norm_stats)
    mean = norm["mean"].astype(np.float32)
    std = norm["std"].astype(np.float32)

    device = torch.device(args.device)

    model = SimpleRNNModel(
    input_dim=512,
    hidden_dim=args.hidden_dim,
    num_classes=41,
    num_layers=args.num_layers,
    bidirectional=args.bidirectional
)

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # support both formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()

    preds_out = []

    with h5py.File(val_h5, "r") as f:
        trial_ids = [k for k in f.keys() if k.startswith("trial_")]
        trial_ids = sorted(trial_ids)

        for tid in trial_ids:
            x = f[tid]["input_features"][()].astype(np.float32)  # (T,512)

            # Apply same normalization as training
            x = (x - mean) / (std + 1e-6)

            xt = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,T,512)
            with torch.no_grad():
                logits = model(xt).squeeze(0)  # (T,C)
                log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

            if args.beam_width <= 1:
                seq = ctc_greedy_decode(log_probs, blank=0)
            else:
                seq = ctc_prefix_beam_search(log_probs, beam_width=args.beam_width, blank=0)

            preds_out.append((tid, np.asarray(seq, dtype=np.int32)))

    np.save(out_dir / "predictions.npy", np.array(preds_out, dtype=object))
    print(f"[OK] wrote {len(preds_out)} predictions to {out_dir / 'predictions.npy'}")


if __name__ == "__main__":
    main()
