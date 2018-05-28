
'''
    file:   mosh_dataloader.py

    author: zhangxiong(1025679612@qq.com)
    date:   2018_05_09
    purpose:  load COCO 2017 keypoint dataset
'''

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
from util import calc_aabb, cut_image, flip_image, draw_lsp_14kp__bone, rectangle_intersect, get_rectangle_intersect_ratio, reflect_pose
from config import args
from timer import Clock


class mosh_dataloader(Dataset):
    def __init__(self, data_set_path, use_flip = True, flip_prob = 0.3):
        self.data_folder = data_set_path
        self.use_flip = use_flip
        self.flip_prob = flip_prob

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
        
        if self.use_flip and random.uniform(0, 1) <= self.flip_prob:#left-right reflect the pose
            pose = reflect_pose(pose)

        return {
            'theta': torch.tensor(np.concatenate((trival, pose, shape), axis = 0)).float()
        }

if __name__ == '__main__':
    print(random.rand(1))
    mosh = mosh_dataloader('E:/HMR/data/mosh_gen')
    l = len(mosh)
    import time
    for _ in range(l):
        r = mosh.__getitem__(_)
        print(r)