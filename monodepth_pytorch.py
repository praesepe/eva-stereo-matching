
# coding: utf-8

# In[130]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
from tensorboardX import SummaryWriter
import argparse
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import import_ipynb
from MonodepthModel import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

torch.cuda.set_device(1)


# In[131]:


def get_args():
    parser = argparse.ArgumentParser(description='Monodepth PyTorch implementation.')
    
    parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
    parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=256)
    parser.add_argument('--input_width',               type=int,   help='input width', default=512)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
    parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
    parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
    parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
    parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
    parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
    parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
    parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
    parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
    parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')
    
    args = parser.parse_args()
    return args


# In[132]:


#args = get_args()
net = MonodepthNet()
optimizer = optim.Adam(net.parameters(), lr=1e-4)


# In[151]:


def get_data(path = '/home/hylai/my/output/folder/'):
    file_path_train = '/home/hylai/monodepth/utils/filenames/kitti_train_files.txt'
    file_path_test = '/home/hylai/monodepth/utils/filenames/kitti_test_files.txt'
    f_train = open(file_path_train)
    f_test = open(file_path_test)
    left_image_train = list()
    right_image_train = list()
    left_image_test = list()
    right_image_test = list()
    for line in f_train:
        left_image_train.append(path+line.split()[0])
        right_image_train.append(path+line.split()[1])
    for line in f_test:
        left_image_test.append(path+line.split()[0])
        right_image_test.append(path+line.split()[1])
    print(left_image_train[0])
    return left_image_train, right_image_train, left_image_test, right_image_test


# In[152]:


def get_transform():
    return transforms.Compose([
        transforms.Scale((256,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.35675976, 0.37380189, 0.3764753), (0.32064945, 0.32098866, 0.32325324))
    ])


# In[153]:


class myImageFolder(data.Dataset):
    def __init__(self, left, right, training):
        self.right = right
        self.left = left
        self.training = training
        
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        left_image = Image.open(left).convert('RGB')
        right_image = Image.open(right).convert('RGB')
        
        #augmentation
        
        #transforms
        process = get_transform()
        left_image = process(left_image)
        right_image = process(right_image)
        
        return left_image, right_image
    def __len__(self):
        return len(self.left)


# In[155]:


#dataloader
left_path = '/home/hylai/my/output/folder/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'
right_path = '/home/hylai/my/output/folder/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/'
left_image = [img for img in os.listdir(left_path)]
right_image = [img for img in os.listdir(right_path)]
img = Image.open(left_path+left_image[30]).convert('RGB')
print(np.array(img).shape)
plt.imshow(img)

left_image_train, right_image_train, left_image_test, right_image_test = get_data()
TrainImageLoader = torch.utils.data.DataLoader(
         myImageFolder(left_image_train, right_image_train, True), 
         batch_size= 1, shuffle= True, num_workers= 1, drop_last=False)
#TestImageLoader

#Test
for batch_idx, (i, j) in enumerate(TrainImageLoader, 0):
    break

