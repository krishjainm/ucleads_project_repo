import h5py

path = r".\data\t15_copyTask_neuralData\hdf5_data_final\t15.2023.10.06\data_train.hdf5"

with h5py.File(path, "r") as f:
    print("TOP LEVEL KEYS:")
    for k in f.keys():
        print(" -", k)
