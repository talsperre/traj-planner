import torch
from ngsim_customloader import ngsimDataset

train_dataset = ngsimDataset(mat_file="../datasets/TrainSet.mat", t_h=32, t_f=48, d_s=4)
# train_dataset = ngsimDataset(mat_file="../datasets/TrainSet.mat")

# print(len(train_dataset))
for i in range(len(train_dataset)):
    hist, fut, neighbors_past, neighbors_fut, lat_enc, lon_enc = train_dataset[i]
    print(hist.shape, fut.shape, len(neighbors_past))
    # for j in range(len(neighbors_past)):
    #     print(neighbors_past[j].shape)
    # print("-"*25)
    # for j in range(len(neighbors_fut)):
    #     print(neighbors_fut[j].shape)
    # print("-"*100)
    # break
    # , neighbors.shape, lat_enc.shape, lon_enc.shape)
    # break