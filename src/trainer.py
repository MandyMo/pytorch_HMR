
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

from util import align_by_pelvis, batch_rodrigues
from timer import Clock
import time
import datetime
from collections import OrderedDict
import os

class HMRTrainer(object):
    def __init__(self):
        self.pix_format = 'NCHW'
        self.normalize = True

        self._build_model()
        self._create_data_loader()

    def _create_data_loader(self):
        self.loader_2d = self._create_2d_data_loader(config.train_2d_set)
        self.loader_mosh = self._create_adv_data_loader(config.train_adv_set)
        self.loader_3d = self._create_3d_data_loader(config.train_3d_set)

    def _build_model(self):
        print('start building modle.')

        self.generator = nn.DataParallel(HMRNetBase()).cuda()
        self.discriminator = nn.DataParallel(Discriminator()).cuda()

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
        self.l2_loss_func = nn.MSELoss()

        print('finished build model.')

    def _create_2d_data_loader(self, data_2d_set):
        data_set = []
        for data_set_name in data_2d_set:
            data_set_path = config.data_set_path[data_set_name]
            if data_set_name == 'coco':
                coco = COCO2017_dataloader(
                    data_set_path = data_set_path, 
                    use_crop = True, 
                    scale_range = [1.1, 1.2], 
                    use_flip = False, 
                    only_single_person = False, 
                    min_pts_required = 7, 
                    max_intersec_ratio = 0.1,
                    pix_format = self.pix_format,
                    normalize = self.normalize
                )
                data_set.append(coco)
            elif data_set_name == 'lsp':
                lsp = LspLoader(
                    data_set_path = data_set_path, 
                    use_crop = True, 
                    scale_range = [1.1, 1.2], 
                    use_flip = False,
                    pix_format = self.pix_format,
                    normalize = self.normalize
                )
                data_set.append(lsp)
            elif data_set_name == 'lsp_ext':
                lsp_ext = LspExtLoader(
                    data_set_path = data_set_path, 
                    use_crop = True, 
                    scale_range = [1.1, 1.2], 
                    use_flip = False,
                    pix_format = self.pix_format,
                    normalize = self.normalize
                )
                data_set.append(lsp_ext)
            elif data_set_name == 'ai-ch':
                ai_ch = AICH_dataloader(
                    data_set_path = data_set_path,
                    use_crop = True, 
                    scale_range = [1.1, 1.2], 
                    use_flip = False, 
                    only_single_person = False, 
                    min_pts_required = 5,
                    max_intersec_ratio = 0.1,
                    pix_format = self.pix_format,
                    normalize = self.normalize
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
                    use_flip = False, 
                    min_pts_required = 5,
                    pix_format = self.pix_format,
                    normalize = self.normalize
                )
                data_set.append(mpi_inf_3dhp)
            elif data_set_name == 'hum3.6m':
                hum36m = hum36m_dataloader(
                    data_set_path = data_set_path, 
                    use_crop = True, 
                    scale_range = [1.1, 1.2], 
                    use_flip = False, 
                    min_pts_required = 5,
                    pix_format = self.pix_format,
                    normalize = self.normalize
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
                    data_set_path = data_set_path
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
            num_workers = config.args.num_worker
        )
    
    def train(self):
        def save_model(save_name):
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
            generator_save_path = os.path.join(parent_folder, 'gen_' + save_name)
            torch.save(exclude_smpl(self.generator.state_dict()), generator_save_path)
            disc_save_path = os.path.join(parent_folder, 'disc_' + save_name)
            torch.save(exclude_smpl(self.discriminator.state_dict()), disc_save_path)

        print('start traing.....')
        torch.backends.cudnn.benchmark = True
        loader_2d, loader_3d, loader_mosh = iter(self.loader_2d), iter(self.loader_3d), iter(self.loader_mosh)
        e_opt, d_opt = self.e_opt, self.d_opt
        self.generator.train() 
        self.discriminator.train()
        for iter_index in range(config.args.iter_count):
            try:
                data_2d = loader_2d.next()
            except StopIteration:
                loader_2d = iter(self.loader_2d)
                data_2d = loader_2d.next()

            try:
                data_3d = loader_3d.next()
            except StopIteration:
                loader_3d = iter(self.loader_3d)
                data_3d = loader_3d.next()
            
            try:
                data_mosh = loader_mosh.next()
            except StopIteration:
                loader_mosh = iter(self.loader_mosh)
                data_mosh = loader_mosh.next()
            
            image_from_2d, image_from_3d = data_2d['image'], data_3d['image']            
            sample_2d_count, sample_3d_count, sample_mosh_count = image_from_2d.shape[0], image_from_3d.shape[0], data_mosh['theta'].shape[0]
            images = torch.cat((image_from_2d, image_from_3d), dim = 0).cuda()

            generator_outputs = self.generator(images)
            if not args.enable_inter_supervision:
                loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, e_disc_loss, d_disc_loss = self._calc_loss(generator_outputs[-1], sample_2d_count, sample_3d_count, sample_mosh_count, data_2d, data_3d, data_mosh)
            else:
                (loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, e_disc_loss, d_disc_loss) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                for generator_output in generator_outputs:
                    c_loss_kp_2d, c_loss_kp_3d, c_loss_shape, c_loss_pose, c_e_disc_loss, c_d_disc_loss = self._calc_loss(generator_output, sample_2d_count, sample_3d_count, sample_mosh_count, data_2d, data_3d, data_mosh)
                    loss_kp_2d += c_loss_kp_2d
                    loss_kp_3d += c_loss_kp_3d
                    loss_shape += c_loss_shape
                    loss_pose += c_loss_shape
                    d_disc_loss += c_d_disc_loss
                    e_disc_loss += c_e_disc_loss
                
                k = len(generator_outputs)
                loss_kp_2d /= k
                loss_kp_3d /= k
                loss_shape /= k
                loss_pose /= k
                d_disc_loss /= k
                e_disc_loss /= k


            e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss
            d_loss = d_disc_loss

            e_opt.zero_grad()
            e_loss.backward(retain_graph=True)
            e_opt.step()

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            iter_msg = OrderedDict(
                [
                    ('time',datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')),
                    ('iter',iter_index),
                    ('e_loss',e_loss.item()),
                    ('2d_loss',loss_kp_2d.item()),
                    ('3d_loss',loss_kp_3d.item()),
                    ('shape_loss',loss_shape.item()),
                    ('pose_loss',loss_pose.item()),
                    ('e_disc_loss',e_disc_loss.item()),
                    ('d_disc_loss',d_disc_loss.item())
                ]
            )

            print(iter_msg)
            
            if iter_index % 500 == 0:
                save_name = str(iter_index) + '_' + str(int(loss_kp_2d.item())) + '_.pkl'
                save_model(save_name)
                print('save model {}'.format(save_name))
            
    
    def _calc_loss(self, generator_output, sample_2d_count, sample_3d_count, sample_mosh_count, data_2d, data_3d, data_mosh):
        (predict_thetas, predict_verts, predict_2d_kp, predict_3d_kp, predict_Rs) = generator_output
        adv_theta = data_mosh['theta'].float().cuda()
        theta_disc_value = self.discriminator(adv_theta, predict_thetas)

        predict_2d_kp, predict_3d_kp = predict_2d_kp[:, :14, :], predict_3d_kp[:, :14, :]
        real_2d_kp = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), dim=0).float().cuda()
        loss_kp_2d = self.batch_kp_2d_l1_loss(real_2d_kp, predict_2d_kp) * args.e_loss_weight
        
        real_3d_kp, predict_3d_kp, w_3d = data_3d['kp_3d'].float().cuda(), predict_3d_kp[sample_2d_count:, :, :], data_3d['w_3d'].float().cuda()
        loss_kp_3d = self.batch_kp_3d_l2_loss(real_3d_kp, predict_3d_kp, w_3d) * args.e_3d_loss_weight

        data_3d_theta = data_3d['theta'].float().cuda()
        real_shape, predict_shape, w_smpl = data_3d_theta[:, 75:], predict_thetas[sample_2d_count:, 75:], data_3d['w_smpl'].float().cuda()
        loss_shape = self.batch_shape_l2_loss(real_shape, predict_shape, w_smpl) * args.e_3d_loss_weight

        real_pose, predict_pose = data_3d_theta[:, 3:75], predict_thetas[sample_2d_count:, 3:75]
        loss_pose = self.batch_pose_l2_loss(real_pose.contiguous(), predict_pose.contiguous(), w_smpl) * args.e_3d_loss_weight

        e_disc_value, d_disc_value = theta_disc_value[sample_mosh_count:, :], theta_disc_value[:sample_mosh_count, :]

        e_disc_loss = self.batch_encoder_disc_l2_loss(e_disc_value) * args.d_loss_weight
        d_disc_loss = self.batch_adv_disc_l2_loss(d_disc_value, e_disc_value) * args.d_loss_weight

        return loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, e_disc_loss, d_disc_loss

    """
        Inputs:
            kp_gt  : N x K x 3
            kp_pred: N x K x 2
    """
    def batch_kp_2d_l1_loss(self, real_2d_kp, predict_2d_kp):
        kp_gt = real_2d_kp.view(-1, 3)
        k = kp_gt.shape[0]
        kp_pred = predict_2d_kp.contiguous().view(-1, 2)
        vis = kp_gt[:, 2]
        dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
        return torch.matmul(dif_abs, vis) * 1.0 / k

    '''
        Inputs:
            real_3d_kp  : N x k x 3
            fake_3d_kp  : N x k x 3
    '''
    def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp, w_3d):
        k = w_3d.shape[0]
        #first align it
        real_3d_kp, fake_3d_kp = align_by_pelvis(real_3d_kp), align_by_pelvis(fake_3d_kp)
        kp_gt = real_3d_kp
        kp_pred = fake_3d_kp
        kp_dif = kp_gt.sub(1.0, kp_pred) ** 2
        return torch.matmul(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k
        
    '''
        Inputs:
            real_shape  :   N x 10
            fake_shape  :   N x 10
    '''
    def batch_shape_l2_loss(self, real_shape, fake_shape, w_shape):
        k = w_shape.shape[0]
        shape_dif = (real_shape - fake_shape) ** 2
        return  torch.matmul(shape_dif.sum(1), w_shape) * 1.0 / k

    '''
        Input:
            real_pose   : N x 72
            fake_pose   : N x 72
    '''
    def batch_pose_l2_loss(self, real_pose, fake_pose, w_pose):
        k = w_pose.shape[0]
        real_rs, fake_rs = batch_rodrigues(real_pose.view(-1, 3)).view(-1, 24, 9)[:, 1:, :], batch_rodrigues(fake_pose.view(-1, 3)).view(-1, 24, 9)[:, 1:, :]
        dif_rs = (real_rs - fake_rs)
        dif_rs = dif_rs * dif_rs
        dif_rs = dif_rs.view(-1, 207)
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
        return torch.sum(fake_disc_value ** 2) / kb + torch.sum((real_disc_value - 1) ** 2) / ka

def main():
    trainer = HMRTrainer()
    trainer.train()

if __name__ == '__main__':
    main()
