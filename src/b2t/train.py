import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np


class BrainTextDataset(Dataset):
    """
    Dataset for HDF5 files structured as:
      trial_0000/
        input_features (T, 512) float32
        seq_class_ids   (L,) int32
        transcription   (L,) int32
    """

    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path

        with h5py.File(hdf5_path, "r") as f:
            self.trials = sorted(
                [k for k in f.keys() if k.startswith("trial_")]
            )

        if len(self.trials) == 0:
            raise RuntimeError(
                f"No trial_* groups found in {hdf5_path}"
            )

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial_key = self.trials[idx]

        with h5py.File(self.hdf5_path, "r") as f:
            g = f[trial_key]
            x = g["input_features"][()]   # (T, 512)
            y = g["seq_class_ids"][()]    # (L,)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        return x, y


class SimpleRNNModel(nn.Module):
    """Minimal RNN model for brain-to-text decoding."""
    
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=41, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.rnn(x)
        # out: (batch, seq_len, hidden_dim)
        logits = self.fc(out)
        # logits: (batch, seq_len, num_classes)
        return logits


def collate_fn(batch):
    """
    Pads input_features (T, 512) to max T in batch.
    For CTC, concatenates targets into 1D vector and returns lengths.
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

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for neural, labels, neural_lengths, label_lengths in dataloader:
        neural = neural.to(device)
        labels = labels.to(device)
        neural_lengths = neural_lengths.to(device)
        label_lengths = label_lengths.to(device)
        
        optimizer.zero_grad()
        
        logits = model(neural)
        # Reshape for CTC: (seq_len, batch, num_classes)
        logits = logits.transpose(0, 1)
        
        # CTC loss expects log_probs
        log_probs = nn.functional.log_softmax(logits, dim=2)
        
        loss = criterion(log_probs, labels, neural_lengths, label_lengths)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Train brain-to-text model')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory containing HDF5 data')
    parser.add_argument('--date', type=str, required=True,
                        help='Date folder name (e.g., t15.2023.08.13)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='RNN hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument(
    '--dry-run',
    action='store_true',
    help='Load one batch and exit (sanity check)'
)
    
    args = parser.parse_args()
    
    # Construct paths
    train_path = os.path.join(args.data_root, args.date, 'data_train.hdf5')
    val_path = os.path.join(args.data_root, args.date, 'data_val.hdf5')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    # Create datasets
    print(f"Loading training data from {train_path}")
    train_dataset = BrainTextDataset(train_path)
    print(f"Training samples: {len(train_dataset)}")
    
    print(f"Loading validation data from {val_path}")
    val_dataset = BrainTextDataset(val_path)
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0 
    )
    if args.dry_run:
        x, y, lx, ly = next(iter(train_loader))
        print("DRY RUN OK")
        print("x shape:", x.shape, x.dtype)
        print("y shape:", y.shape, y.dtype)
        print("x lengths:", lx[:5].tolist())
        print("y lengths:", ly[:5].tolist())
        return

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    sample_neural, _ = train_dataset[0]
    input_dim = sample_neural.shape[-1] 
    
    device = torch.device(args.device)
    model = SimpleRNNModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=41,  # From config: n_classes=41
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'final_checkpoint.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
