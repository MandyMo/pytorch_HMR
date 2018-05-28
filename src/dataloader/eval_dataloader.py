
'''
    file:   eval_dataloader.py

    author: zhangxiong(1025679612@qq.com)
    date:   2018_05_20
    purpose:  load evaluation data
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
import torch

sys.path.append('./src')
from util import calc_aabb, cut_image, flip_image, draw_lsp_14kp__bone, rectangle_intersect, get_rectangle_intersect_ratio, convert_image_by_pixformat_normalize, reflect_pose
from config import args
# from timer import Clock

class eval_dataloader(Dataset):
    def __init__(self, data_set_path, use_flip, pix_format = 'NHWC', normalize = False, flip_prob = 0.3):
        self.use_flip    = use_flip
        self.flip_prob   = flip_prob
        self.data_folder = data_set_path
        self.pix_format  = pix_format
        self.normalize   = normalize

        self._load_data_set()

    def _load_data_set(self):
        # clk = Clock()
        
        self.images = sorted(glob.glob(os.path.join(self.data_folder, 'image/*.png')))
        self.kp2ds = []
        self.poses = []
        self.betas = []

        for idx in range(len(self.images)):
            image_name = os.path.basename(self.images[idx])[:5]
            anno_path = os.path.join(self.data_folder, 'annos', image_name + '_joints.npy')
            self.kp2ds.append(np.load(anno_path).T)
            anno_path = os.path.join(self.data_folder, 'annos', image_name +'.json')
            with open(anno_path, 'r') as fp:
                annos = json.load(fp)
                self.poses.append(np.array(annos['pose']))
                self.betas.append(np.array(annos['betas']))

        # clk.stop()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        pose = self.poses[index].copy()
        shape = self.betas[index].copy()

        dst_image = cv2.imread(image_path)


        if self.use_flip and random.random() <= self.flip_prob:
            dst_image, kps = flip_image(dst_image, kps)
            pose = reflect_poses(pose)
    
        #normalize kp to [-1, 1]
        ratio = 1.0 / args.crop_size
        kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0

        return {
            'image': torch.tensor(convert_image_by_pixformat_normalize(dst_image, self.pix_format, self.normalize)).float(),
            'kp_2d': torch.tensor(kps).float(),
            'pose': torch.tensor(pose).float(),
            'shape': torch.tensor(shape).float(),
            'image_name': self.images[index],
            'data_set':'up_3d_evaluation'
        }

if __name__ == '__main__':
    evl = eval_dataloader('E:/HMR/data/up3d_mpii', True)
    l = evl.__len__()
    data_loader = DataLoader(evl, batch_size=10,shuffle=True)
    for _ in range(l):
        r = evl.__getitem__(_)
        pass
        
    
