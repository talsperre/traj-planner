from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):
    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13,3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,8:]
        neighbors_past = []
        neighbors_fut = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)
        fut = self.getFuture(vehId, t, vehId, dsId)

        # Get track histories and future track of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors_past.append(self.getHistory(i.astype(int), t, vehId, dsId))
            neighbors_fut.append(self.getFuture(i.astype(int), t, vehId, dsId))

        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 7] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 6] - 1)] = 1

        return hist, fut, neighbors_past, neighbors_fut, lat_enc, lon_enc

    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1] <= vehId-1:
                return np.empty([0, 2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist = hist[1:]

            if len(hist) < self.t_h // self.d_s:
                return np.empty([0,2])
            return hist

    ## Helper function to get track history
    def getFuture(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                 return np.empty([0, 2])
            else:
                stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
                enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
                fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
            return fut
