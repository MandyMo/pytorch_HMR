
'''
    file:   lsp_dataloader.py

    author: zhangxiong(1025679612@qq.com)
    date:   2018_05_07
'''
import sys
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio  
import os 
import glob
import numpy as np
import random
import cv2
import torch

sys.path.append('./src')

from util import calc_aabb, cut_image, flip_image, draw_lsp_14kp__bone, convert_image_by_pixformat_normalize, reflect_lsp_kp
from config import args
from timer import Clock

class LspLoader(Dataset):
    def __init__(self, data_set_path, use_crop, scale_range, use_flip, pix_format = 'NHWC', normalize = False, flip_prob = 0.3):
        '''
            marks:
                data_set path links to the parent folder to lsp, which contains images, joints.mat, README.txt
            
            inputs:
                use_crop crop the image or not, it should be True by default
                scale_range, contain the scale range
                use_flip, left right flip is allowed
        '''
        self.use_crop    = use_crop
        self.scale_range = scale_range
        self.use_flip    = use_flip
        self.flip_prob   = flip_prob
        self.data_folder = data_set_path
        self.pix_format  = pix_format
        self.normalize   = normalize

        self._load_data_set()

    def _load_data_set(self):
        clk = Clock()
        print('loading LSP data.')
        self.images = []
        self.kp2ds  = []
        self.boxs   = []

        anno_file_path = os.path.join(self.data_folder, 'joints.mat')
        anno = scio.loadmat(anno_file_path)
        kp2d = anno['joints'].transpose(2, 1, 0) # N x k x 3
        visible = np.logical_not(kp2d[:, :, 2])
        kp2d[:, :, 2] = visible.astype(kp2d.dtype)
        image_folder = os.path.join(self.data_folder, 'images')
        images = sorted(glob.glob(image_folder + '/im*.jpg'))
        for _ in range(len(images)):
            self._handle_image(images[_], kp2d[_])

        print('finished load LSP data.')
        clk.stop()
        
    def _handle_image(self, image_path, kps):
        pt_valid = []
        for pt in kps:
            if pt[2] == 1:
                pt_valid.append(pt)
        lt, rb, valid = calc_aabb(pt_valid)

        if not valid:
            return

        self.kp2ds.append(kps.copy().astype(np.float))
        self.images.append(image_path)
        self.boxs.append((lt, rb))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        box = self.boxs[index]

        scale = np.random.rand(4) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        image, kps = cut_image(image_path, kps, scale, box[0], box[1])
        ratio = 1.0 * args.crop_size / image.shape[0]
        kps[:, :2] *= ratio
        dst_image = cv2.resize(image, (args.crop_size, args.crop_size), interpolation = cv2.INTER_CUBIC)

        if self.use_flip and random.random() <= self.flip_prob:
            dst_image, kps = flip_image(dst_image, kps)
        
        #normalize kp to [-1, 1]
        ratio = 1.0 / args.crop_size
        kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0
        return {
            'image': torch.tensor(convert_image_by_pixformat_normalize(dst_image, self.pix_format, self.normalize)).float(),
            'kp_2d': torch.tensor(kps).float(),
            'image_name': self.images[index],
            'data_set':'lsp'
        }

if __name__ == '__main__':
    lsp = LspLoader(
        data_set_path = 'E:/HMR/data/lsp', 
        use_crop = True, 
        scale_range = [1.05, 1.2], 
        use_flip = True,
        flip_prob = 1.0
    )
    l = lsp.__len__()
    data_loader = DataLoader(lsp, batch_size=10,shuffle=True)
    for _ in range(l):
        r = lsp.__getitem__(_)
        image = r['image'].cpu().numpy().astype(np.uint8)
        kps = r['kp_2d'].cpu().numpy()
        kps[:, :2] = (kps[:, :2] + 1) * args.crop_size / 2.0
        base_name = os.path.basename(r['image_name'])
        draw_lsp_14kp__bone(image, kps)
        cv2.imshow(base_name, cv2.resize(image, (512, 512), interpolation = cv2.INTER_CUBIC))
        cv2.waitKey(0)
    
