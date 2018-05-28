
'''
    file:   COCO2017_dataloader.py

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
import torch

sys.path.append('./src')
from util import calc_aabb, cut_image, flip_image, draw_lsp_14kp__bone, rectangle_intersect, get_rectangle_intersect_ratio, convert_image_by_pixformat_normalize
from config import args
from timer import Clock

class COCO2017_dataloader(Dataset):
    def __init__(self, data_set_path, use_crop, scale_range, use_flip, only_single_person, min_pts_required, max_intersec_ratio = 0.1, pix_format = 'NHWC', normalize = False, flip_prob = 0.3):
        self.data_folder = data_set_path
        self.use_crop = use_crop
        self.scale_range = scale_range
        self.use_flip = use_flip
        self.flip_prob = flip_prob
        self.only_single_person = only_single_person
        self.min_pts_required = min_pts_required
        self.max_intersec_ratio = max_intersec_ratio
        self.pix_format = pix_format
        self.normalize = normalize
        self._load_data_set()

    def _load_data_set(self):
        self.images = []
        self.kp2ds  = []
        self.boxs   = []
        clk = Clock()
        print('start loading coco 2017 dataset.')
        anno_file_path = os.path.join(self.data_folder, 'annotations', 'person_keypoints_train2017.json')
        with open(anno_file_path, 'r') as reader:
            anno = json.load(reader)
        
        def _hash_image_id_(image_id_to_info, coco_images_info):
            for image_info in coco_images_info:
                image_id = image_info['id']
                image_name = image_info['file_name']
                _anno = {}
                _anno['image_path'] = os.path.join(self.data_folder, 'images', 'train-valid2017', image_name)
                _anno['kps'] = []
                _anno['box'] = []
                assert not (image_id in image_id_to_info)
                image_id_to_info[image_id] = _anno
                
        images = anno['images']

        image_id_to_info = {}
        _hash_image_id_(image_id_to_info, images)


        annos = anno['annotations']
        for anno_info in annos:
            self._handle_anno_info(anno_info, image_id_to_info)

        for k, v in image_id_to_info.items():
            self._handle_image_info_(v)

        print('finished load coco 2017 dataset, total {} samples.'.format(len(self.images)))

        clk.stop()
        
    def _handle_image_info_(self, image_info):
        image_path = image_info['image_path']
        kp_set = image_info['kps']
        box_set = image_info['box']
        if len(box_set) > 1:
            if self.only_single_person:
                return

        for _ in range(len(box_set)):
            self._handle_sample(_, kp_set, box_set, image_path)

    def _handle_sample(self, key, kps, boxs, image_path):
        def _collect_box(l, boxs):
            r = []
            for _ in range(len(boxs)):
                if _ == l:
                    continue
                r.append(boxs[_])
            return r

        def _collide_heavily(box, boxs):
            for it in boxs:
                if get_rectangle_intersect_ratio(box[0], box[1], it[0], it[1]) > self.max_intersec_ratio:
                    return True
            return False

        kp = kps[key]
        box = boxs[key]

        valid_pt_cound = np.sum(kp[:, 2])
        if valid_pt_cound < self.min_pts_required:
            return

        r = _collect_box(key, boxs)
        if _collide_heavily(box, r):
            return
        
        self.images.append(image_path)
        self.kp2ds.append(kp.copy())
        self.boxs.append(box.copy())

    def _handle_anno_info(self, anno_info, image_id_to_info):
        image_id = anno_info['image_id']
        kps = anno_info['keypoints']
        box_info = anno_info['bbox']
        box = [np.array([int(box_info[0]), int(box_info[1])]), np.array([int(box_info[0] + box_info[2]), int(box_info[1] + box_info[3])])]
        assert image_id in image_id_to_info
        _anno = image_id_to_info[image_id]
        _anno['box'].append(box)
        _anno['kps'].append(self._convert_to_lsp14_pts(kps))

    def _convert_to_lsp14_pts(self, coco_pts):
        kp_map = [15, 13, 11, 10, 12, 14, 9, 7, 5, 4, 6, 8, 0, 0]
        kp_map = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, 0, 0]
        kps = np.array(coco_pts, dtype = np.float).reshape(-1, 3)[kp_map].copy()
        kps[12: ,2] = 0.0 #no neck, top head
        kps[:, 2] /= 2.0
        return kps

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
            'data_set':'COCO 2017'
        }

if __name__ == '__main__':
    coco = COCO2017_dataloader('E:/HMR/data/COCO/', True, [1.1, 1.5], False, False, 10, 0.1)
    l = len(coco)
    for _ in range(l):
        r = lsp.__getitem__(_)
        image = r['image'].cpu().numpy().astype(np.uint8)
        kps = r['kp_2d'].cpu().numpy()
        base_name = os.path.basename(r['image_name'])
        draw_lsp_14kp__bone(image, kps)
        cv2.imshow(base_name, cv2.resize(image, (512, 512), interpolation = cv2.INTER_CUBIC))
        cv2.waitKey(0)
