
from LinearModel import LinearModel
import config
import util
import torch
import numpy as np
import torch.nn as nn
from config import args
import torch.nn.functional as F

class ShapeDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)

        super(ShapeDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)
    
    def forward(self, inputs):
        return self.fc_blocks(inputs)

class PoseDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)
        
        super(PoseDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        '''
        x = self.fc_blocks(inputs)
        return [x, self.last_block(x)]
        '''
        return self.fc_blocks(inputs)

class FullPoseDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)

        super(FullPoseDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        return self.fc_blocks(inputs)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self._read_configs()

        self._create_sub_modules()

    def _read_configs(self):
        self.beta_count = args.beta_count
        self.smpl_model = args.smpl_model
        self.smpl_mean_theta_path = args.smpl_mean_theta_path
        self.total_theta_count = args.total_theta_count
        self.joint_count = args.joint_count
        self.feature_count = args.feature_count

    def _create_sub_modules(self):
        '''
            create theta discriminator for 23 joint
        '''
        fc_layers = [9, 32, 32, 1]
        use_dropout = [False, False, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        self.pose_discriminators = nn.ModuleList()
        for _ in range(self.joint_count - 1):
            self.pose_discriminators.append(PoseDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func))
        
        '''
            create full pose discriminator for total 23 joints
        '''
        fc_layers = [(self.joint_count - 1) * 9, 1024, 1024, 1024, 1]
        use_dropout = [False, False, False, False]
        drop_prob = [0.5, 0.5, 0.5, 0.5]
        use_ac_func = [True, True, True, False]
        self.full_pose_discriminator = FullPoseDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func)

        '''
            shape discriminator for betas
        '''
        fc_layers = [self.beta_count, 5, 1]
        use_dropout = [False, False]
        drop_prob = [0.5, 0.5]
        use_ac_func = [True, False]
        self.shape_discriminator = ShapeDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func)

        print('finished create the discriminator modules...')

    '''
        purpose:
            calc mean shape discriminator value
        inputs:
            real_shape N x 10
            fake_shape n x 10
        return:
            shape discriminator output value
    '''
    def calc_shape_disc_value(self, real_shape, fake_shape):
        shapes = torch.cat([real_shape, fake_shape], dim = 0)
        return self.shape_discriminator(shapes)

    '''
        inputs:
            real_pose N x 24 x 3
            fake_pose n x 24 x 3
        return:
            pose discriminator output value
    '''
    def calc_pose_disc_value(self, real_pose, fake_pose):
        real_pose = util.batch_rodrigues(real_pose.view(-1, 3)).view(-1, 24, 9)
        fake_pose = util.batch_rodrigues(fake_pose.view(-1, 3)).view(-1, 24, 9)
        poses = torch.cat((real_pose[:, 1:, :], fake_pose[:, 1:, :]), dim = 0)
        full_pose_dis_value = self.full_pose_discriminator(poses.view(-1, 23 * 9))
        poses = torch.transpose(poses, 0, 1)
        theta_disc_values = []
        for _ in range(23):
            theta_disc_values.append(
                self.pose_discriminators[_](poses[_, :, :])
            )
        pose_dis_value = torch.cat(theta_disc_values, dim = 1)
        return torch.cat([pose_dis_value, full_pose_dis_value], dim = 1)        

    '''
        inputs:
            real_thetas N x 85
            fake_thetas N x 85
        return
            pose & full pose & shape disc value N x (23 + 1 + 1)
    '''
    def calc_thetas_disc_value(self, real_thetas, fake_thetas):
        real_poses, fake_poses = real_thetas[:, 3:75], fake_thetas[:, 3:75]
        real_shapes, fake_shapes = real_thetas[:, 75:], fake_thetas[:, 75:]
        pose_disc_value = self.calc_pose_disc_value(real_poses.contiguous(), fake_poses.contiguous())
        shape_disc_value = self.calc_shape_disc_value(real_shapes.contiguous(), fake_shapes.contiguous())
        return torch.cat([pose_disc_value, shape_disc_value], dim = 1)

    def forward(self, real_thetas, fake_thetas):
        if config.args.normalize_disc:
            return F.sigmoid(self.calc_thetas_disc_value(real_thetas, fake_thetas))
        else:
            return self.calc_thetas_disc_value(real_thetas, fake_thetas)


if __name__ == '__main__':
    device = torch.device('cuda')
    net = Discriminator().to(device)
    real = torch.zeros((100, 85)).float().to(device)
    fake = torch.ones((200, 85)).float().to(device)

    dis_v = net(real, fake)
    print(dis_v.device)
    print(dis_v.shape)
