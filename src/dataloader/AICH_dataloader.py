

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
from util import calc_aabb, cut_image, flip_image, draw_lsp_14kp__bone, rectangle_intersect, get_rectangle_intersect_ratio, convert_image_by_pixformat_normalize
from config import args
from timer import Clock

class AICH_dataloader(Dataset):
    def __init__(self, data_set_path, use_crop, scale_range, use_flip, only_single_person, min_pts_required, max_intersec_ratio = 0.1, pix_format = 'NHWC', normalize = False):
        self.data_folder = data_set_path
        self.use_crop = use_crop
        self.scale_range = scale_range
        self.use_flip = use_flip
        self.only_single_person = only_single_person
        self.min_pts_required = min_pts_required
        self.max_intersec_ratio = max_intersec_ratio
        self.img_ext = '.jpg'
        self.pix_format = pix_format
        self.normalize = normalize
        self._load_data_set()
    
    def _load_data_set(self):
        clk = Clock()

        self.images = []
        self.kp2ds  = []
        self.boxs   = []
        print('start loading AI CH keypoint data.')
        anno_file_path = os.path.join(self.data_folder, 'keypoint_train_annotations_20170902.json')
        with open(anno_file_path, 'r') as reader:
            anno = json.load(reader)
        for record in anno:
            image_name = record['image_id'] + self.img_ext
            image_path = os.path.join(self.data_folder, 'keypoint_train_images_20170902', image_name)
            kp_set = record['keypoint_annotations']
            box_set = record['human_annotations']
            self._handle_image(image_path, kp_set, box_set)

        print('finished load Ai CH keypoint data, total {} samples'.format(len(self)))

        clk.stop()

    def _ai_ch_to_lsp(self, pts):
        kp_map = [8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5, 13, 12]
        pts = np.array(pts, dtype = np.float).reshape(14, 3).copy()
        pts[:, 2] = (3.0 - pts[:, 2]) / 2.0
        return pts[kp_map].copy()

    def _handle_image(self, image_path, kp_set, box_set):
        assert len(kp_set) == len(box_set)

        if len(kp_set) > 1:
            if self.only_single_person:
                print('only single person supported now!')
                return
        for key in kp_set.keys():
            kps = kp_set[key]
            box = box_set[key]
            self._handle_sample(key, image_path, kps, [ [box[0], box[1]], [box[2], box[3]] ], box_set)

    def _handle_sample(self, key, image_path, pts, box, boxs):
        def _collect_box(key, boxs):
            r = []
            for k, v in boxs.items():
                if k == key:
                    continue
                r.append([[v[0],v[1]], [v[2],v[3]]])
            return r

        def _collide_heavily(box, boxs):
            for it in boxs:
                if get_rectangle_intersect_ratio(box[0], box[1], it[0], it[1]) > self.max_intersec_ratio:
                    return True
            return False
        pts = self._ai_ch_to_lsp(pts)
        valid_pt_cound = np.sum(pts[:, 2])
        if valid_pt_cound < self.min_pts_required:
            return

        boxs = _collect_box(key, boxs)
        if _collide_heavily(box, boxs):
            return

        self.images.append(image_path)
        self.kp2ds.append(pts)
        lt, rb = box[0], box[1]
        self.boxs.append((np.array(lt), np.array(rb)))

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
            dst_image = flip_image(dst_image, kps, random.randint(-1, 1))
            
        return {
            'image': torch.from_numpy(convert_image_by_pixformat_normalize(dst_image, self.pix_format, self.normalize)).float(),
            'kp_2d': torch.from_numpy(kps).float(),
            'image_name': self.images[index],
            'data_set':'AI Ch'
        }

if __name__ == '__main__':
    aic = AICH_dataloader('E:/HMR/data/ai_challenger_keypoint_train_20170902', True, [1.1, 1.5], False, False, 5)
    l = len(aic)
    for _ in range(l):
        r = aic.__getitem__(_)
        base_name = os.path.basename(r['image_name'])
        draw_lsp_14kp__bone(r['image'], r['kp_2d'])
        cv2.imshow(base_name, cv2.resize(r['image'], (512, 512), interpolation = cv2.INTER_CUBIC))
        cv2.waitKey(0)
