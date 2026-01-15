import h5py
import numpy as np

path = r".\data\t15_copyTask_neuralData\hdf5_data_final\t15.2023.10.06\data_train.hdf5"
trial = "trial_0083"

with h5py.File(path, "r") as f:
    g = f[trial]
    for name in ["input_features", "seq_class_ids", "transcription"]:
        d = g[name]
        print("\n", name)
        print("  shape:", d.shape)
        print("  dtype:", d.dtype)
        # show a tiny preview safely
        try:
            x = d[()]
            if isinstance(x, (bytes, np.bytes_)):
                print("  preview:", x[:80])
            elif hasattr(x, "shape"):
                print("  preview:", x.flatten()[:10])
            else:
                print("  preview:", x)
        except Exception as e:
            print("  preview error:", e)
