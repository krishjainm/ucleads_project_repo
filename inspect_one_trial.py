import h5py

path = r".\data\t15_copyTask_neuralData\hdf5_data_final\t15.2023.10.06\data_train.hdf5"
trial = "trial_0083"   # pick any one you saw in the list

with h5py.File(path, "r") as f:
    g = f[trial]
    print("TRIAL:", trial)
    print("Keys under trial:")
    for k in g.keys():
        print(" -", k)
