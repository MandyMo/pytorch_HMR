
'''
    file:   Discriminator.py

    date:   2017_04_29
    author: zhangxiong(1025679612@qq.com)
'''

from LinearModel import LinearModel
import config
import util
import torch
import numpy as np
import torch.nn as nn
from config import args

'''
    shape discriminator is used for shape discriminator
    the inputs if N x 10
'''
class ShapeDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)

        super(ShapeDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)
    
    def forward(self, inputs):
        return self.fc_blocks(inputs)

class PoseDiscriminator(nn.Module):
    def __init__(self, channels):
        super(PoseDiscriminator, self).__init__()

        if channels[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(channels[-1])
            sys.exit(msg)
        
        self.conv_blocks = nn.Sequential()
        l = len(channels)
        for idx in range(l - 2):
            self.conv_blocks.add_module(
                name = 'conv_{}'.format(idx),
                module = nn.Conv2d(in_channels = channels[idx], out_channels = channels[idx + 1], kernel_size = 1, stride = 1)
            )

        self.fc_layer = nn.ModuleList()
        for idx in range(23):
            self.fc_layer.append(nn.Linear(in_features = channels[l - 2], out_features = 1))

    # N x 23 x 9
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(1, 2).unsqueeze(2) # to N x 9 x 1 x 23
        internal_outputs = self.conv_blocks(inputs) # to N x c x 1 x 23
        o = []
        for idx in range(23):
            o.append(self.fc_layer[idx](internal_outputs[:,:,0,idx]))
        
        return torch.cat(o, 1), internal_outputs

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

        self.pose_discriminator = PoseDiscriminator([9, 32, 32, 1])
        
        '''
            create full pose discriminator for total 23 joints
        '''
        fc_layers = [23 * 32, 1024, 1024, 1]
        use_dropout = [False, False, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
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
        inputs is N x 85(3 + 72 + 10)
    '''
    def forward(self, thetas):
        batch_size = thetas.shape[0]
        cams, poses, shapes = thetas[:, :3], thetas[:, 3:75], thetas[:, 75:]
        shape_disc_value = self.shape_discriminator(shapes)
        rotate_matrixs = util.batch_rodrigues(poses.contiguous().view(-1, 3)).view(-1, 24, 9)[:, 1:, :]
        pose_disc_value, pose_inter_disc_value = self.pose_discriminator(rotate_matrixs)
        full_pose_disc_value = self.full_pose_discriminator(pose_inter_disc_value.contiguous().view(batch_size, -1))
        return torch.cat((pose_disc_value, full_pose_disc_value, shape_disc_value), 1)

if __name__ == '__main__':
    device = torch.device('cuda')
    net = Discriminator()
    inputs = torch.ones((100, 85))
    disc_value = net(inputs)
    print(net)