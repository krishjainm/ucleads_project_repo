import numpy as np
import h5py

preds = np.load("outputs/ablations/gru2_hd512_bs8_e10/predictions.npy", allow_pickle=True)

with h5py.File("data/t15_copyTask_neuralData/hdf5_data_final/t15.2023.10.06/data_val.hdf5", "r") as f:
    for i in range(3):
        tid, pred = preds[i]
        gt = f[tid]["transcription"][:].astype(int)

        print("="*60)
        print("Trial:", tid)
        print("Pred (first 50):", pred[:50])
        print("GT   (first 50):", gt[:50])
