# Phase 1 — Data Forensics (HDF5)

All scripts are designed to run on **Windows** from the **project root** and only use: `numpy`, `pandas`, `matplotlib`, `h5py`, `argparse`, `os`, `pathlib`.

They read:
- `data_train.hdf5`
- `data_val.hdf5`

from a specific session folder (date), with HDF5 structure:
- top-level groups `trial_####`
  - `input_features` \((T, 512)\) `float32`
  - `seq_class_ids` \((500,)\) `int32`
  - `transcription` \((500,)\) `int32` (ASCII codes; padded with 0)

All outputs are written to `figures/phase1/` (created automatically).

## How to run

Use the same CLI shape for every script:

```bash
python analysis/inspect_dataset.py --data-root ".\data\t15_copyTask_neuralData\hdf5_data_final" --date "t15.2023.10.06"
python analysis/plot_trial_structure.py --data-root ".\data\t15_copyTask_neuralData\hdf5_data_final" --date "t15.2023.10.06"
python analysis/plot_neural_stats.py --data-root ".\data\t15_copyTask_neuralData\hdf5_data_final" --date "t15.2023.10.06"
```

## Outputs

- `dataset_summary.csv`: counts + example trial keys + feature-dim (512) violations summary.
- `trial_lengths.csv`: per-trial `neural_len`, `label_len`, and `transcript_len` table.
- `hist_neural_len.(png|pdf)`: distribution of input sequence lengths \(T\).
- `hist_label_len.(png|pdf)`: distribution of non-pad label lengths (assumes `seq_class_ids == 0` is padding).
- `scatter_neural_len_vs_label_len.(png|pdf)`: relationship between neural length and label length.
- `channel_mean_var.csv`: per-channel mean/variance for `input_features` over a reproducible sample (up to 200 trials, seed=0).
- `channel_variance.(png|pdf)`: variance across the 512 channels.
- `channel_correlation_heatmap.(png|pdf)`: channel correlation heatmap from a limited subsample of timepoints.
- `temporal_energy.csv`: mean/std temporal energy curve vs time index.
- `temporal_energy_curve.(png|pdf)`: average temporal energy curve with a ±1 std band.

