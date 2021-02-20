from __future__ import print_function, division

import os
import torch
import numpy as np
import scipy.io as scp
import matplotlib.pyplot as plt

from PIL import Image
from utils import gkern
from torch.utils.data import Dataset, DataLoader


### Dataset class for the NGSIM dataset
class inferDataset(Dataset):
    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.D = self.D[self.D[:, 2] % 25 == 0]
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid
        self.lat_size = 240 # Each pixel is 0.1m in vertical direction
        self.lon_size = 360 # Each pixel is 0.5m in horizontal direction
        self.lat_centre = 110
        self.lon_centre = 120
        self.lane_pos = np.array([-1.5, 75.0, 11.5, 23.5, 35.0, 48.0, 60.0])
        self.num_past = int(self.t_h / self.d_s)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,8:]
        neighbors = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)
        fut, refPos = self.getFuture(vehId, t, dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId))

        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 7] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 6] - 1)] = 1
        
        ego_occ_grid, veh_occ_grid = self.generate_infer_data(hist, neighbors, refPos)
        fut_occ_grid = np.zeros((self.lat_size, self.lon_size))
        self.gen_occ_grid(fut_occ_grid, fut[-1], refPos)
        return ego_occ_grid, veh_occ_grid, fut_occ_grid, lat_enc, lon_enc
    
    ## Helper function to get track history
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
                hist = hist[1:]
            if len(hist) < self.t_h//self.d_s:
                return np.empty([0,2])
            return hist

    ## Helper function to get track future
    def getFuture(self, vehId, t,dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut, refPos
    
    def generate_infer_data(self, hist, neighbors, ref_pos):
        ego_occ_grid = np.zeros((self.num_past, self.lat_size, self.lon_size))
        veh_occ_grid = np.zeros((self.num_past, self.lat_size, self.lon_size))
        self.add_lanes(veh_occ_grid)
        self.add_lanes(ego_occ_grid)
        
        for i in range(self.num_past - 1, -1, -1):
            self.gen_occ_grid(ego_occ_grid[i], hist[i], ref_pos)
        
        for i in range(len(neighbors)):
            if len(neighbors[i]):
                occ_idx = self.num_past - 1
                for j in range(neighbors[i].shape[0] - 1, -1, -1):
                    self.gen_occ_grid(veh_occ_grid[occ_idx], neighbors[i][j], ref_pos)
                    occ_idx -= 1
        return ego_occ_grid, veh_occ_grid
    
    def add_lanes(self, occ_grid):
        # Need to convert feet to m
        lanes_lat = 12 - self.lane_pos * 0.3048
        lanes_lat = (self.lat_centre + np.around(lanes_lat * 10)).astype(int)
        occ_grid[:, lanes_lat, :] = 1
    
    def gen_occ_grid(self, occ_grid, point, ref_pos):
        # Need to convert feet to m
        lat = 12 - (point[0] + ref_pos[0]) * 0.3048 
        lon = point[1] * 0.3048

        ego_lat = (self.lat_centre + np.around(lat * 10)).astype(int)
        ego_lon = (self.lon_centre + np.around(lon * 2)).astype(int)

        # gauss = gkern(9, 2.25)
        # gauss = gkern(12, 2.25)
        # for k, lat in enumerate(range(ego_lat - 4, ego_lat + 5)):
        #     for l, lon in enumerate(range(ego_lon - 4, ego_lon + 5)):
        #         if lat >= 0 and lat < 240 and lon >= 0 and lon < 360:
        #             occ_grid[lat, lon] = gauss[k, l]
        gauss = gkern(21, 2)
        for k, lat_pos in enumerate(range(ego_lat - 10, ego_lat + 11)):
            for l, lon_pos in enumerate(range(ego_lon - 10, ego_lon + 11)):
                if lat_pos >= 0 and lat_pos < 240 and lon_pos >= 0 and lon_pos < 360:
                    occ_grid[lat_pos, lon_pos] = gauss[k, l]        
