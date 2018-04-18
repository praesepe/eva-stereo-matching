
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
from tensorboardX import SummaryWriter
import random
import os
import os.path
import matplotlib.image as mpimg
from PIL import Image
import import_ipynb
from MonodepthModel import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

torch.cuda.set_device(1)


# In[2]:


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


# In[3]:


#args = get_args()
net = MonodepthNet().cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-4)


# In[4]:


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


# In[5]:


def get_transform():
    return transforms.Compose([
        transforms.Scale([512, 256]),
        transforms.ToTensor()
    ])


# In[6]:


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
        if not self.training:
            
            #randomly flip
            if random.uniform(0, 1) > 0.5:
                left_image = left_image.transpose(Image.FLIP_LEFT_RIGHT)
                right_image = right_image.transpose(Image.FLIP_LEFT_RIGHT)
                
            #randomly shift gamma
            if random.uniform(0, 1) > 0.5:
                gamma = random.uniform(0.8, 1.2)
                left_image = Image.fromarray(np.clip((np.array(left_image) ** gamma), 0, 255).astype('uint8'), 'RGB')
                right_image = Image.fromarray(np.clip((np.array(right_image) ** gamma), 0, 255).astype('uint8'), 'RGB')
            
            #randomly shift brightness
            if random.uniform(0, 1) > 0.5:
                brightness = random.uniform(0.5, 2.0)
                left_image = Image.fromarray(np.clip((np.array(left_image) * brightness), 0, 255).astype('uint8'), 'RGB')
                right_image = Image.fromarray(np.clip((np.array(right_image) * brightness), 0, 255).astype('uint8'), 'RGB')
            
            #randomly shift color
            if random.uniform(0, 1) > 0.5:
                colors = [random.uniform(0.8, 1.2) for i in range(3)]
                shape = np.array(left_image).shape
                white = np.ones((shape[0], shape[1]))
                color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
                left_image = Image.fromarray(np.clip((np.array(left_image) * color_image), 0, 255).astype('uint8'), 'RGB')
                right_image = Image.fromarray(np.clip((np.array(right_image) * color_image), 0, 255).astype('uint8'), 'RGB')
                
        
        #transforms
        process = get_transform()
        left_image = process(left_image)
        right_image = process(right_image)
        
        return left_image, right_image
    def __len__(self):
        return len(self.left)


# In[7]:


def make_pyramid(image, num_scales):
    scale_image = [image]
    height, width = image.shape[2:]

    for i in range(num_scales - 1):
        new = []
        for j in range(image.shape[0]):
            ratio = 2 ** (i+1)
            nh = height // ratio
            nw = width // ratio
            tmp = transforms.ToPILImage()(image[j]).convert('RGB')
            tmp = transforms.Scale([nw, nh])(tmp)
            new.append(np.array(tmp))
        scale_image.append(new)
        
    return scale_image


# In[8]:


def gradient_x(img):
    print(img[:,:,:-1,:].shape, img[:,:,1:,:].shape)
    gx = torch.add(img[:,:,:-1,:], -1, img[:,:,1:,:])
    return gx

def gradient_y(img):
    gy = torch.add(img[:,:,:,:-1], -1, img[:,:,:,1:])
    return gy

def get_disparity_smoothness(disp, pyramid):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    #image_gradients_x = [gradient_x(img) for img in pyramid]
    #image_gradients_y = [gradient_y(img) for img in pyramid]
    
    return 1


# In[9]:


left_image_train, right_image_train, left_image_test, right_image_test = get_data()
TrainImageLoader = torch.utils.data.DataLoader(
         myImageFolder(left_image_train, right_image_train, True), 
         batch_size = 8, shuffle = True, num_workers = 8, drop_last =False)
TestImageLoader = torch.utils.data.DataLoader(
         myImageFolder(left_image_test, right_image_test, False),
         batch_size = 8, shuffle = False, num_workers = 8, drop_last =False)

#Train
do_stereo = 0
for epoch in range(1):
    for batch_idx, (left, right) in enumerate(TestImageLoader, 0):

        #generate image pyramid[scale][batch]
        left_pyramid = make_pyramid(left, 4)
        right_pyramid = make_pyramid(right, 4)
        
        if do_stereo:
            model_input = Variable(torch.cat((left, right), 0).cuda())
        else:
            model_input = Variable(left.cuda())
            
        disp_est = net(model_input)
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in disp_est]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in disp_est]
        #x = disp_left_est[0][0,0,:,:].data.cpu().numpy()
        #plt.imshow(x)
        
        #generate image
        #left_est = [warp(right_pyramid[i], disp_left_est[i]) for i in range(4)]
        #right_est = [warp(left_pyramid[i], disp_right_est[i]) for i in range(4)]
        
        #LR consistency
        #right_to_left_disp = [warp(disp_right_est[i], disp_left_est[i]) for i in range(4)]
        #left_to_right_disp = [warp(disp_left_est[i], disp_right_est[i]) for i in range(4)]
        
        #disparity smoothness
        #disp_left_smoothness = get_disparity_smoothness(disp_left_est, left_pyramid)
        #disp_right_smoothness = get_disparity_smoothness(disp_right_est, right_pyramid)
    
        break

