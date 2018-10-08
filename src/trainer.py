

'''
    file:   trainer.py

    date:   2018_05_07
    author: zhangxiong(1025679612@qq.com)
'''

import sys
from model import HMRNetBase
from Discriminator import Discriminator
from config import args
import config
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from dataloader.AICH_dataloader import AICH_dataloader
from dataloader.COCO2017_dataloader import COCO2017_dataloader
from dataloader.hum36m_dataloader import hum36m_dataloader
from dataloader.lsp_dataloader import LspLoader
from dataloader.lsp_ext_dataloader import LspExtLoader
from dataloader.mosh_dataloader import mosh_dataloader
from dataloader.mpi_inf_3dhp_dataloader import mpi_inf_3dhp_dataloader
from dataloader.eval_dataloader import eval_dataloader

from util import align_by_pelvis, batch_rodrigues, copy_state_dict
from timer import Clock
import time
import datetime
from collections import OrderedDict
import os

class HMRTrainer(object):
    def __init__(self):
        self.pix_format = 'NCHW'
        self.normalize = True
        self.flip_prob = 0.5
        self.use_flip = False
        self.w_smpl = torch.ones((config.args.eval_batch_size)).float().cuda()

        self._build_model()
        self._create_data_loader()

    def _create_data_loader(self):
        self.loader_2d = self._create_2d_data_loader(config.train_2d_set)
        self.loader_mosh = self._create_adv_data_loader(config.train_adv_set)
        self.loader_3d = self._create_3d_data_loader(config.train_3d_set)
        
    def _build_model(self):
        print('start building modle.')

        '''
            load pretrain model
        '''
        generator = HMRNetBase()
        model_path = config.pre_trained_model['generator']
        if os.path.exists(model_path):
            copy_state_dict(
                generator.state_dict(), 
                torch.load(model_path),
                prefix = 'module.'
            )
        else:
            print('model {} not exist!'.format(model_path))

        discriminator = Discriminator()
        model_path = config.pre_trained_model['discriminator']
        if os.path.exists(model_path):
            copy_state_dict(
                discriminator.state_dict(),
                torch.load(model_path),
                prefix = 'module.'
            )
        else:
            print('model {} not exist!'.format(model_path))

        self.generator = nn.DataParallel(generator).cuda()
        self.discriminator = nn.DataParallel(discriminator).cuda()
        
        self.e_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr = args.e_lr,
            weight_decay = args.e_wd
        )
    
        self.d_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr = args.d_lr,
            weight_decay = args.d_wd
        )

        self.e_sche = torch.optim.lr_scheduler.StepLR(
            self.e_opt,
            step_size = 500,
            gamma = 0.9
        )

        self.d_sche = torch.optim.lr_scheduler.StepLR(
            self.d_opt,
            step_size = 500,
            gamma = 0.9
        )

        print('finished build model.')

    def _create_2d_data_loader(self, data_2d_set):
        data_set = []
        for data_set_name in data_2d_set:
            data_set_path = config.data_set_path[data_set_name]
            if data_set_name == 'coco':
                coco = COCO2017_dataloader(
                    data_set_path = data_set_path, 
                    use_crop = True, 
                    scale_range = [1.05, 1.3], 
                    use_flip = self.use_flip, 
                    only_single_person = False, 
                    min_pts_required = 7, 
                    max_intersec_ratio = 0.5,
                    pix_format = self.pix_format,
                    normalize = self.normalize,
                    flip_prob = self.flip_prob
                )
                data_set.append(coco)
            elif data_set_name == 'lsp':
                lsp = LspLoader(
                    data_set_path = data_set_path, 
                    use_crop = True, 
                    scale_range = [1.05, 1.3], 
                    use_flip = self.use_flip,
                    pix_format = self.pix_format,
                    normalize = self.normalize,
                    flip_prob = self.flip_prob
                )
                data_set.append(lsp)
            elif data_set_name == 'lsp_ext':
                lsp_ext = LspExtLoader(
                    data_set_path = data_set_path, 
                    use_crop = True, 
                    scale_range = [1.1, 1.2], 
                    use_flip = self.use_flip,
                    pix_format = self.pix_format,
                    normalize = self.normalize,
                    flip_prob = self.flip_prob
                )
                data_set.append(lsp_ext)
            elif data_set_name == 'ai-ch':
                ai_ch = AICH_dataloader(
                    data_set_path = data_set_path,
                    use_crop = True, 
                    scale_range = [1.1, 1.2], 
                    use_flip = self.use_flip, 
                    only_single_person = False, 
                    min_pts_required = 5,
                    max_intersec_ratio = 0.1,
                    pix_format = self.pix_format,
                    normalize = self.normalize,
                    flip_prob = self.flip_prob
                )
                data_set.append(ai_ch)
            else:
                msg = 'invalid 2d dataset'
                sys.exit(msg)

        con_2d_dataset = ConcatDataset(data_set)

        return DataLoader(
            dataset = con_2d_dataset,
            batch_size = config.args.batch_size,
            shuffle = True,
            drop_last = True,
            pin_memory = True,
            num_workers = config.args.num_worker
        )

    def _create_3d_data_loader(self, data_3d_set):
        data_set = []
        for data_set_name in data_3d_set:
            data_set_path = config.data_set_path[data_set_name]
            if data_set_name == 'mpi-inf-3dhp':
                mpi_inf_3dhp = mpi_inf_3dhp_dataloader(
                    data_set_path = data_set_path, 
                    use_crop = True, 
                    scale_range = [1.1, 1.2], 
                    use_flip = self.use_flip, 
                    min_pts_required = 5,
                    pix_format = self.pix_format,
                    normalize = self.normalize,
                    flip_prob = self.flip_prob
                )
                data_set.append(mpi_inf_3dhp)
            elif data_set_name == 'hum3.6m':
                hum36m = hum36m_dataloader(
                    data_set_path = data_set_path, 
                    use_crop = True, 
                    scale_range = [1.1, 1.2], 
                    use_flip = self.use_flip, 
                    min_pts_required = 5,
                    pix_format = self.pix_format,
                    normalize = self.normalize,
                    flip_prob = self.flip_prob
                )
                data_set.append(hum36m)
            else:
                msg = 'invalid 3d dataset'
                sys.exit(msg)

        con_3d_dataset = ConcatDataset(data_set)

        return DataLoader(
            dataset = con_3d_dataset,
            batch_size = config.args.batch_3d_size,
            shuffle = True,
            drop_last = True,
            pin_memory = True,
            num_workers = config.args.num_worker
        )
    
    def _create_adv_data_loader(self, data_adv_set):
        data_set = []
        for data_set_name in data_adv_set:
            data_set_path = config.data_set_path[data_set_name]
            if data_set_name == 'mosh':
                mosh = mosh_dataloader(
                    data_set_path = data_set_path,
                    use_flip = self.use_flip,
                    flip_prob = self.flip_prob
                )
                data_set.append(mosh)
            else:
                msg = 'invalid adv dataset'
                sys.exit(msg)

        con_adv_dataset = ConcatDataset(data_set)
        return DataLoader(
            dataset = con_adv_dataset,
            batch_size = config.args.adv_batch_size,
            shuffle = True,
            drop_last = True,
            pin_memory = True,
        )
    
    def _create_eval_data_loader(self, data_eval_set):
        data_set = []
        for data_set_name in data_eval_set:
            data_set_path = config.data_set_path[data_set_name]
            if data_set_name == 'up3d':
                up3d = eval_dataloader(
                    data_set_path = data_set_path,
                    use_flip = False,
                    flip_prob = self.flip_prob,
                    pix_format = self.pix_format,
                    normalize = self.normalize
                )
                data_set.append(up3d)
            else:
                msg = 'invalid eval dataset'
                sys.exit(msg)
        con_eval_dataset = ConcatDataset(data_set)
        return DataLoader(
            dataset = con_eval_dataset,
            batch_size = config.args.eval_batch_size,
            shuffle = False,
            drop_last = False,
            pin_memory = True,
            num_workers = config.args.num_worker
        )

    def train(self):
        def save_model(result):
            exclude_key = 'module.smpl'
            def exclude_smpl(model_dict):
                result = OrderedDict()
                for (k, v) in model_dict.items():
                    if exclude_key in k:
                        continue
                    result[k] = v
                return result

            parent_folder = args.save_folder
            if not os.path.exists(parent_folder):
                os.makedirs(parent_folder)

            title = result['title']
            generator_save_path = os.path.join(parent_folder, title + 'generator.pkl')
            torch.save(exclude_smpl(self.generator.state_dict()), generator_save_path)
            disc_save_path = os.path.join(parent_folder, title + 'discriminator.pkl')
            torch.save(exclude_smpl(self.discriminator.state_dict()), disc_save_path)
            with open(os.path.join(parent_folder, title + '.txt'), 'w') as fp:
                fp.write(str(result))
        
        #pre_best_loss = None

        torch.backends.cudnn.benchmark = True
        loader_2d, loader_3d, loader_mosh = iter(self.loader_2d), iter(self.loader_3d), iter(self.loader_mosh)
        e_opt, d_opt = self.e_opt, self.d_opt
        
        self.generator.train()
        self.discriminator.train()

        for iter_index in range(config.args.iter_count):
            try:
                data_2d = next(loader_2d)
            except StopIteration:
                loader_2d = iter(self.loader_2d)
                data_2d = next(loader_2d)

            try:
                data_3d = next(loader_3d)
            except StopIteration:
                loader_3d = iter(self.loader_3d)
                data_3d = next(loader_3d)
            
            try:
                data_mosh = next(loader_mosh)
            except StopIteration:
                loader_mosh = iter(self.loader_mosh)
                data_mosh = next(loader_mosh)
            
            image_from_2d, image_from_3d = data_2d['image'], data_3d['image']            
            sample_2d_count, sample_3d_count, sample_mosh_count = image_from_2d.shape[0], image_from_3d.shape[0], data_mosh['theta'].shape[0]
            images = torch.cat((image_from_2d, image_from_3d), dim = 0).cuda()

            generator_outputs = self.generator(images)

            loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, data_2d, data_3d, data_mosh)
            
            e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss
            d_loss = d_disc_loss

            e_opt.zero_grad()
            e_loss.backward()
            e_opt.step()

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            loss_kp_2d = float(loss_kp_2d)
            loss_shape = float(loss_shape / args.e_shape_ratio)
            loss_kp_3d = float(loss_kp_3d / args.e_3d_kp_ratio)
            loss_pose  = float(loss_pose / args.e_pose_ratio)
            e_disc_loss = float(e_disc_loss / args.d_disc_ratio)
            d_disc_loss = float(d_disc_loss / args.d_disc_ratio)

            d_disc_real = float(d_disc_real / args.d_disc_ratio)
            d_disc_predict = float(d_disc_predict / args.d_disc_ratio)

            e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss
            d_loss = d_disc_loss
        
            iter_msg = OrderedDict(
                [
                    ('time',datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')),
                    ('iter',iter_index),
                    ('e_loss', e_loss),
                    ('2d_loss',loss_kp_2d),
                    ('3d_loss',loss_kp_3d),
                    ('shape_loss',loss_shape),
                    ('pose_loss', loss_pose),
                    ('e_disc_loss',float(e_disc_loss)),
                    ('d_disc_loss',float(d_disc_loss)),
                    ('d_disc_real', float(d_disc_real)),
                    ('d_disc_predict', float(d_disc_predict))
                ]
            )

            print(iter_msg)            

            if iter_index % 500 == 0:
                iter_msg['title'] = '{}_{}_'.format(iter_msg['iter'], iter_msg['e_loss'])
                save_model(iter_msg)

    def _calc_loss(self, generator_outputs, data_2d, data_3d, data_mosh):
        def _accumulate_thetas(generator_outputs):
            thetas = []
            for (theta, verts, j2d, j3d, Rs) in generator_outputs:
                thetas.append(theta)
            return torch.cat(thetas, 0)

        sample_2d_count, sample_3d_count, sample_mosh_count = data_2d['kp_2d'].shape[0], data_3d['kp_2d'].shape[0], data_mosh['theta'].shape
        data_3d_theta, w_3d, w_smpl = data_3d['theta'].cuda(), data_3d['w_3d'].float().cuda(), data_3d['w_smpl'].float().cuda()

        total_predict_thetas = _accumulate_thetas(generator_outputs)
        (predict_theta, predict_verts, predict_j2d, predict_j3d, predict_Rs) = generator_outputs[-1]

        real_2d, real_3d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0).cuda(), data_3d['kp_3d'].float().cuda()
        predict_j2d, predict_j3d, predict_theta = predict_j2d, predict_j3d[sample_2d_count:, :], predict_theta[sample_2d_count:, :]

        loss_kp_2d = self.batch_kp_2d_l1_loss(real_2d, predict_j2d[:,:14,:]) *  args.e_loss_weight
        loss_kp_3d = self.batch_kp_3d_l2_loss(real_3d, predict_j3d[:,:14,:], w_3d) * args.e_3d_loss_weight * args.e_3d_kp_ratio
        
        real_shape, predict_shape = data_3d_theta[:, 75:], predict_theta[:, 75:]
        loss_shape = self.batch_shape_l2_loss(real_shape, predict_shape, w_smpl) * args.e_3d_loss_weight * args.e_shape_ratio

        real_pose, predict_pose = data_3d_theta[:, 3:75], predict_theta[:, 3:75]
        loss_pose = self.batch_pose_l2_loss(real_pose.contiguous(), predict_pose.contiguous(), w_smpl) * args.e_3d_loss_weight * args.e_pose_ratio

        e_disc_loss = self.batch_encoder_disc_l2_loss(self.discriminator(total_predict_thetas)) * args.d_loss_weight * args.d_disc_ratio
        
        mosh_real_thetas = data_mosh['theta'].cuda()
        fake_thetas = total_predict_thetas.detach()
        fake_disc_value, real_disc_value = self.discriminator(fake_thetas), self.discriminator(mosh_real_thetas)
        d_disc_real, d_disc_fake, d_disc_loss = self.batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)
        d_disc_real, d_disc_fake, d_disc_loss = d_disc_real  * args.d_loss_weight * args.d_disc_ratio, d_disc_fake  * args.d_loss_weight * args.d_disc_ratio, d_disc_loss * args.d_loss_weight * args.d_disc_ratio

        return loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, e_disc_loss, d_disc_loss, d_disc_real, d_disc_fake

    """
        purpose:
            calc L1 error
        Inputs:
            kp_gt  : N x K x 3
            kp_pred: N x K x 2
    """
    def batch_kp_2d_l1_loss(self, real_2d_kp, predict_2d_kp):
        kp_gt = real_2d_kp.view(-1, 3)
        kp_pred = predict_2d_kp.contiguous().view(-1, 2)
        vis = kp_gt[:, 2]
        k = torch.sum(vis) * 2.0 + 1e-8
        dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
        return torch.matmul(dif_abs, vis) * 1.0 / k

    '''
        purpose:
            calc mse * 0.5

        Inputs:
            real_3d_kp  : N x k x 3
            fake_3d_kp  : N x k x 3
            w_3d        : N x 1
    '''
    def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp, w_3d):
        shape = real_3d_kp.shape
        k = torch.sum(w_3d) * shape[1] * 3.0 * 2.0 + 1e-8

        #first align it
        real_3d_kp, fake_3d_kp = align_by_pelvis(real_3d_kp), align_by_pelvis(fake_3d_kp)
        kp_gt = real_3d_kp
        kp_pred = fake_3d_kp
        kp_dif = (kp_gt - kp_pred) ** 2
        return torch.matmul(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k
        
    '''
        purpose:
            calc mse * 0.5

        Inputs:
            real_shape  :   N x 10
            fake_shape  :   N x 10
            w_shape     :   N x 1
    '''
    def batch_shape_l2_loss(self, real_shape, fake_shape, w_shape):
        k = torch.sum(w_shape) * 10.0 * 2.0 + 1e-8
        shape_dif = (real_shape - fake_shape) ** 2
        return  torch.matmul(shape_dif.sum(1), w_shape) * 1.0 / k

    '''
        Input:
            real_pose   : N x 72
            fake_pose   : N x 72
    '''
    def batch_pose_l2_loss(self, real_pose, fake_pose, w_pose):
        k = torch.sum(w_pose) * 207.0 * 2.0 + 1e-8
        real_rs, fake_rs = batch_rodrigues(real_pose.view(-1, 3)).view(-1, 24, 9)[:,1:,:], batch_rodrigues(fake_pose.view(-1, 3)).view(-1, 24, 9)[:,1:,:]
        dif_rs = ((real_rs - fake_rs) ** 2).view(-1, 207)
        return torch.matmul(dif_rs.sum(1), w_pose) * 1.0 / k
    '''
        Inputs:
            disc_value: N x 25
    '''
    def batch_encoder_disc_l2_loss(self, disc_value):
        k = disc_value.shape[0]
        return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k
    '''
        Inputs:
            disc_value: N x 25
    '''
    def batch_adv_disc_l2_loss(self, real_disc_value, fake_disc_value):
        ka = real_disc_value.shape[0]
        kb = fake_disc_value.shape[0]
        lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
        return la, lb, la + lb

def main():
    trainer = HMRTrainer()
    trainer.train()

if __name__ == '__main__':
    main()
