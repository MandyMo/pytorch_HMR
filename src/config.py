
import argparse

parser = argparse.ArgumentParser(description = 'hmr model')

parser.add_argument(
    '--encoder-network',
    type = str,
    default = 'hourglass',
    help = 'the encoder network name'
)

parser.add_argument(
    '--enable-inter-supervision',
    type = bool,
    default = True,
    help = 'use intermidiate supervision or not'
)

parser.add_argument(
    '--smpl-mean-theta-path', 
    type = str, 
    default = '/media/disk1/projects/HMR_hourglass/model/neutral_smpl_mean_params.h5', 
    help = 'the path for mean smpl theta value'
)

parser.add_argument(
    '--smpl-model',
    type = str,
    default = '/media/disk1/projects/HMR_hourglass/model/neutral_smpl_with_cocoplus_reg.txt',
    help = 'smpl model path'
)

parser.add_argument(
    '--total-theta-count', 
    type = int, 
    default = 85,
    help = 'the count of theta param'
)

parser.add_argument(
    '--batch-size',
    type = int,
    default = 16,
    help = 'batch size'
)

parser.add_argument(
    '--batch-3d-size',
    type = int,
    default = 16,
    help = '3d data batch size'
)

parser.add_argument(
    '--adv-batch-size',
    type = int,
    default = 32,
    help = 'default adv batch size'
)

parser.add_argument(
    '--joint-count',
    type = int,
    default = 24,
    help = 'the count of joints'
)

parser.add_argument(
    '--beta-count',
    type = int,
    default = 10,
    help = 'the count of beta'
)

parser.add_argument(
    '--feature-count',
    type = int,
    default = 4096,
    help = 'the count of feature count (for resnet is 2048, hourglass is 4096)'
)

parser.add_argument(
    '--use-adv-train',
    type = bool,
    default = True,
    help = 'use adv traing or not'
)

parser.add_argument(
    '--crop-size',
    type = int,
    default = 256,
    help = 'croped image size'
)

parser.add_argument(
    '--scale-min',
    type = float,
    default = 1.1,
    help = 'min scale'
)

parser.add_argument(
    '--scale-max',
    type = float,
    default = 1.5,
    help = 'max scale'
)

parser.add_argument(
    '--normalize-disc',
    type = bool,
    default = True,
    help = 'use sigmoid to normalize the output of discvalue or not'
)

parser.add_argument(
    '--num-worker',
    type = int,
    default = 2,
    help = 'pytorch number worker.'
)

parser.add_argument(
    '--iter-count',
    type = int,
    default = 500001,
    help = 'iter count, eatch contains batch-size samples'
)

parser.add_argument(
    '--e-lr',
    type = float,
    default = 0.001,
    help = 'encoder learning rate.'
)

parser.add_argument(
    '--d-lr',
    type = float,
    default = 0.001,
    help = 'Adversarial prior learning rate.'
)

parser.add_argument(
    '--e-wd',
    type = float,
    default = 0.0001,
    help = 'encoder weight decay rate.'
)

parser.add_argument(
    '--d-wd',
    type = float,
    default = 0.0001,
    help = 'Adversarial prior weight decay'
)

parser.add_argument(
    '--e-loss-weight', 
    type = float,
    default = 60, 
    help = 'weight on encoder 2d kp losses.'
)

parser.add_argument(
    '--d-loss-weight',
    type = float,
    default = 1,
    help = 'weight on discriminator losses'
)

parser.add_argument(
    '--e-3d-loss-weight',
    type = float,
    default = 1,
    help = 'weight on encoder thetas losses.'
)

parser.add_argument(
    '--save-folder',
    type = str,
    default = '/media/disk1/projects/HMR_hourglass/trained_model',
    help = 'save model path'
)

train_2d_set = ['coco', 'lsp', 'lsp_ext', 'ai-ch']
train_3d_set = ['mpi-inf-3dhp', 'hum3.6m']
train_adv_set = ['mosh']

allowed_encoder_net = ['hourglass', 'resnet50']

encoder_feature_count = {
    'hourglass' : 4096,
    'resnet50' : 2048
}

data_set_path = {
    'coco':'/media/disk1/database/COCO/',
    'lsp':'/media/disk1/database/lsp',
    'lsp_ext':'/media/disk1/database/lsp_ext',
    'ai-ch':'/media/disk1/database/ai_challenger_keypoint_train_20170902',
    'mpi-inf-3dhp':'/media/disk1/database/mpi_inf_3dhp',
    'hum3.6m':'/media/disk1/database/human3.6m',
    'mosh':'/media/disk1/database/mosh_gen'
}

args = parser.parse_args()
