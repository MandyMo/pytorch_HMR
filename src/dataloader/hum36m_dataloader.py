

import sys
from torch.utils.data import Dataset, DataLoader
import os 
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch

sys.path.append('./src')
from util import calc_aabb, cut_image, flip_image, draw_lsp_14kp__bone, rectangle_intersect, get_rectangle_intersect_ratio, convert_image_by_pixformat_normalize
from config import args
from timer import Clock

class hum36m_dataloader(Dataset):
    def __init__(self, data_set_path, use_crop, scale_range, use_flip, min_pts_required, pix_format = 'NHWC', normalize = False):
        self.data_folder = data_set_path
        self.use_crop = use_crop
        self.scale_range = scale_range
        self.use_flip = use_flip
        self.min_pts_required = min_pts_required
        self.pix_format = pix_format
        self.normalize = normalize
        self._load_data_set()

    def _load_data_set(self):
        clk = Clock()

        self.images = []
        self.kp2ds  = []
        self.boxs   = []
        self.kp3ds  = []
        self.shapes = []
        self.poses  = []

        print('start loading hum3.6m data.')

        anno_file_path = os.path.join(self.data_folder, 'annot.h5')
        with h5py.File(anno_file_path) as fp:
            total_kp2d = np.array(fp['gt2d'])
            total_kp3d = np.array(fp['gt3d'])
            total_shap = np.array(fp['shape'])
            total_pose = np.array(fp['pose'])
            total_image_names = np.array(fp['imagename'])

            assert len(total_kp2d) == len(total_kp3d) and len(total_kp2d) == len(total_image_names) and \
                len(total_kp2d) == len(total_shap) and len(total_kp2d) == len(total_pose)

            l = len(total_kp2d)
            def _collect_valid_pts(pts):
                r = []
                for pt in pts:
                    if pt[2] != 0:
                        r.append(pt)
                return r

            for index in range(l):
                kp2d = total_kp2d[index].reshape((-1, 3))
                if np.sum(kp2d[:, 2]) < self.min_pts_required:
                    continue
                
                lt, rb, v = calc_aabb(_collect_valid_pts(kp2d))
                self.kp2ds.append(np.array(kp2d.copy(), dtype = np.float))
                self.boxs.append((lt, rb))
                self.kp3ds.append(total_kp3d[index].copy().reshape(-1, 3))
                self.shapes.append(total_shap[index].copy())
                self.poses.append(total_pose[index].copy())
                self.images.append(os.path.join(self.data_folder, 'image') + total_image_names[index].decode())

        print('finished load hum3.6m data, total {} samples'.format(len(self.kp3ds)))
        
        clk.stop()

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        box = self.boxs[index]

        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        image, kps = cut_image(image_path, kps, scale, box[0], box[1])

        ratio = 1.0 * args.crop_size / image.shape[0]
        kps[:, 0] *= ratio
        kps[:, 1] *= ratio
        dst_image = cv2.resize(image, (args.crop_size, args.crop_size), interpolation = cv2.INTER_CUBIC)

        if self.use_flip:
            assert 0
            dst_image = flip_image(dst_image, kps, random.randint(-1, 1))
        
        trival, pose, shape = np.zeros(3), self.shapes[index], self.poses[index]

        theta = np.concatenate((trival, pose, shape), axis = 0)

        return {
            'image': torch.from_numpy(convert_image_by_pixformat_normalize(dst_image, self.pix_format, self.normalize)).float(),
            'kp_2d': torch.from_numpy(kps).float(),
            'kp_3d': torch.from_numpy(self.kp3ds[index]).float(),
            'theta': torch.from_numpy(theta).float(),
            'image_name': self.images[index],
            'w_smpl':1.0,
            'w_3d':1.0,
            'data_set':'hum3.6m'
        }

if __name__ == '__main__':
    h36m = hum36m_dataloader('E:/HMR/data/human3.6m', True, [1.1, 2.0], False, 5)
    l = len(h36m)
    for _ in range(l):
        r = h36m.__getitem__(_)
        base_name = os.path.basename(r['image_name'])
        draw_lsp_14kp__bone(r['image'], r['kp_2d'])
        cv2.imshow(base_name, cv2.resize(r['image'], (512, 512), interpolation = cv2.INTER_CUBIC))
        cv2.waitKey(0)
