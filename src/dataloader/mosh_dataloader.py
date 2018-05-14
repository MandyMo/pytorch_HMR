

import sys
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio  
import os 
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch

sys.path.append('./src')
from util import calc_aabb, cut_image, flip_image, draw_lsp_14kp__bone, rectangle_intersect, get_rectangle_intersect_ratio
from config import args
from timer import Clock


class mosh_dataloader(Dataset):
    def __init__(self, data_set_path):
        self.data_folder = data_set_path
        self._load_data_set()

    def _load_data_set(self):
        clk = Clock()
        print('start loading mosh data.')
        anno_file_path = os.path.join(self.data_folder, 'mosh_annot.h5')
        with h5py.File(anno_file_path) as fp:
            self.shapes = np.array(fp['shape'])
            self.poses = np.array(fp['pose'])
        print('finished load mosh data, total {} samples'.format(len(self.poses)))
        clk.stop()

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        trival, pose, shape = np.zeros(3), self.poses[index], self.shapes[index]
        return {
            'theta': np.concatenate((trival, pose, shape), axis = 0)
        }

if __name__ == '__main__':
    mosh = mosh_dataloader('E:/HMR/data/mosh_gen')
    l = len(mosh)
    import time
    for _ in range(l):
        r = mosh.__getitem__(_)
        print(r)
