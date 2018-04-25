
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
import skimage.transform
from tensorboardX import SummaryWriter
import random
import os
import os.path
import matplotlib.image as mpimg
from PIL import Image
import import_ipynb
from MonodepthModel import *
import scipy.misc
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
#net = MonodepthNet().cuda()
net = torch.load("/home/hylai/monodepth/model_city2kitti")
optimizer = optim.Adam(net.parameters(), lr=1e-4)
params = list(net.parameters())
name = list(net.named_parameters())


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
    
    num_train = 0
    num_test = 0
    
    for line in f_train:
        num_train += 1
        left_image_train.append(path+line.split()[0])
        right_image_train.append(path+line.split()[1])
    for line in f_test:
        num_test += 1
        left_image_test.append(path+line.split()[0])
        right_image_test.append(path+line.split()[1])
        
    return left_image_train, right_image_train, left_image_test, right_image_test, num_train, num_test


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
        if self.training:
            
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
    scale_image = [Variable(image.cuda())]
    height, width = image.shape[2:]

    for i in range(num_scales - 1):
        new = []
        for j in range(image.shape[0]):
            ratio = 2 ** (i+1)
            nh = height // ratio
            nw = width // ratio
            tmp = transforms.ToPILImage()(image[j]).convert('RGB')
            tmp = transforms.Scale([nw, nh])(tmp)
            tmp = transforms.ToTensor()(tmp)
            new.append(tmp.unsqueeze(0))
        this = torch.cat((i for i in new), 0)
        scale_image.append(Variable(this.cuda()))
        
    return scale_image


# In[8]:


def gradient_x(img):
    gx = torch.add(img[:,:,:-1,:], -1, img[:,:,1:,:])
    return gx

def gradient_y(img):
    gy = torch.add(img[:,:,:,:-1], -1, img[:,:,:,1:])
    return gy

def get_disparity_smoothness(disp, pyramid):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]
    
    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]
    
    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
    
    return smoothness_x + smoothness_y


# In[9]:


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    
    #(input, kernel, stride, padding)
    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 1) - mu_x * mu_y
    
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    SSIM = SSIM_n / SSIM_d
    
    return torch.clamp((1 - SSIM) / 2, 0, 1)


# In[10]:


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


# In[29]:


def wrap(input_images, x_offset, wrap_mode='border'):
    num_batch, num_channel, height, width = input_images.shape
    width_f = float(width)
    height_f = float(height)
    x_t, y_t = np.meshgrid(np.linspace(0.0, width_f-1, width), np.linspace(0.0, height_f-1, height))

    x_t_flat = np.reshape(x_t, (1, -1))
    y_t_flat = np.reshape(y_t, (1, -1))

    x_t_flat = np.tile(x_t_flat, (num_batch, 1))
    y_t_flat = np.tile(y_t_flat, (num_batch, 1))

    x_t_flat = np.reshape(x_t_flat, (-1))
    y_t_flat = np.reshape(y_t_flat, (-1))
    
    y_t_flat = Variable(torch.FloatTensor(y_t_flat).cuda())
    x_t_flat = Variable(torch.FloatTensor(x_t_flat).cuda())
    
    x_t_flat = x_t_flat + x_offset.view(-1) * width_f
    
    #interpolate
    edge_size = 1
    input_images = nn.ZeroPad2d(1)(input_images)
    x = x_t_flat + edge_size
    y = y_t_flat + edge_size
    
    x = torch.clamp(x, 0.0, width_f - 1 + 2 * edge_size)
    x0_f = torch.floor(x)
    y0_f = torch.floor(y)
    x1_f = x0_f + 1
    
    x0 = Variable(x0_f.data.type(torch.IntTensor).cuda())
    y0 = Variable(y0_f.data.type(torch.IntTensor).cuda())
    x1 = Variable(torch.clamp(x1_f.data, 0.0, width_f - 1 + 2 * edge_size).type(torch.IntTensor).cuda())
    
    dim2 = width + 2 * edge_size
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
    base = Variable((torch.arange(num_batch)*dim1).unsqueeze(1).repeat(1, height * width).view(-1).type(torch.IntTensor).cuda())
    base_y0 = base + y0 * dim2
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1
    
    im_flat = input_images.permute(0,2,3,1).contiguous().view(-1, num_channel)
    pix_l = torch.index_select(im_flat, 0, idx_l.type(torch.LongTensor).cuda())
    pix_r = torch.index_select(im_flat, 0, idx_r.type(torch.LongTensor).cuda())
    
    weight_l = (x1_f - x).unsqueeze(1)
    weight_r = (x - x0_f).unsqueeze(1)
    
    return (weight_l * pix_l + weight_r * pix_r).view(num_batch, height, width, num_channel).permute(0,3,1,2)


# In[57]:


left_image_train, right_image_train, left_image_test, right_image_test, num_train, num_test = get_data()
TrainImageLoader = torch.utils.data.DataLoader(
         myImageFolder(left_image_train, right_image_train, True), 
         batch_size = 8, shuffle = True, num_workers = 8, drop_last =False)
TestImageLoader = torch.utils.data.DataLoader(
         myImageFolder(left_image_test, right_image_test, False),
         batch_size = 8, shuffle = False, num_workers = 8, drop_last =False)

#Train
do_stereo = 0
alpha_image_loss = 0.85
disp_gradient_loss_weight = 0.1
lr_loss_weight = 1.0
num_epochs = 10
optimizer = optim.Adam(net.parameters(), lr = 0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32((3/5) * num_epochs), np.int32((4/5) * num_epochs)], gamma=0.5)

for epoch in range(1):
    for batch_idx, (left, right) in enumerate(TrainImageLoader, 0):

        #generate image pyramid[scale][batch]
        left_pyramid = make_pyramid(left, 4)
        right_pyramid = make_pyramid(right, 4)
        
        if do_stereo:
            model_input = Variable(torch.cat((left, right), 1).cuda())
        else:
            model_input = Variable(left.cuda())
        
        disp_est = net(model_input)
        disp_left_est = [d[:, 0, :, :].contiguous().unsqueeze(1) for d in disp_est]
        disp_right_est = [d[:, 1, :, :].contiguous().unsqueeze(1) for d in disp_est]
        
        #generate image
        left_est = [wrap(right_pyramid[i], -disp_left_est[i]) for i in range(4)]
        right_est = [wrap(left_pyramid[i], disp_right_est[i]) for i in range(4)]
        a = left_est[0][0,:,:,:].data.cpu().numpy().transpose((1,2,0))
        plt.imshow(a)
        
        #LR consistency
        right_to_left_disp = [wrap(disp_right_est[i], -disp_left_est[i]) for i in range(4)]
        left_to_right_disp = [wrap(disp_left_est[i], disp_right_est[i]) for i in range(4)]
        
        #disparity smoothness
        disp_left_smoothness = get_disparity_smoothness(disp_left_est, left_pyramid)
        disp_right_smoothness = get_disparity_smoothness(disp_right_est, right_pyramid)
        
        #build loss
        #L1 loss
        l1_left = [torch.abs(left_est[i] - left_pyramid[i]) for i in range(4)]
        l1_reconstruction_loss_left  = [torch.mean(l) for l in l1_left]
        l1_right = [torch.abs(right_est[i] - right_pyramid[i]) for i in range(4)]
        l1_reconstruction_loss_right  = [torch.mean(l) for l in l1_right]
        
        #SSIM
        ssim_left = [SSIM(left_est[i], left_pyramid[i]) for i in range(4)]
        ssim_loss_left = [torch.mean(s) for s in ssim_left]
        ssim_right = [SSIM(right_est[i], right_pyramid[i]) for i in range(4)]
        ssim_loss_right = [torch.mean(s) for s in ssim_right]
        
        #Weighted Sum
        image_loss_right = [alpha_image_loss * ssim_loss_right[i] + (1 - alpha_image_loss) * l1_reconstruction_loss_right[i] for i in range(4)]
        image_loss_left  = [alpha_image_loss * ssim_loss_left[i]  + (1 - alpha_image_loss) * l1_reconstruction_loss_left[i]  for i in range(4)]
        image_loss = np.sum(image_loss_left + image_loss_right)
        
        #Disparity smoothness
        disp_left_loss  = [torch.mean(torch.abs(disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
        disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(4)]
        disp_gradient_loss = np.sum(disp_left_loss + disp_right_loss)
        
        #LR consistency
        lr_left_loss  = [torch.mean(torch.abs(right_to_left_disp[i] - disp_left_est[i]))  for i in range(4)]
        lr_right_loss = [torch.mean(torch.abs(left_to_right_disp[i] - disp_right_est[i])) for i in range(4)]
        lr_loss = np.sum(lr_left_loss + lr_right_loss)
        
        #Total loss
        total_loss = image_loss + disp_gradient_loss_weight * disp_gradient_loss + lr_loss_weight * lr_loss
        break


# In[56]:


#test
import scipy.misc

input_image = scipy.misc.imread("./test.jpg", mode="RGB")
original_height, original_width, num_channels = input_image.shape
input_image = scipy.misc.imresize(input_image, [256, 512], interp='lanczos')
input_image = input_image.astype(np.float32) / 255
input_images = np.stack((input_image, np.fliplr(input_image)), 0)
model_input = Variable(torch.from_numpy(input_images.transpose((0,3,1,2))).cuda())
disp_est = net(model_input)


disp_pp = post_process_disparity(disp_est[0][:,0,:,:].data.cpu().numpy())
disp_to_img = scipy.misc.imresize(disp_pp, [original_height, original_width])
plt.imshow(disp_to_img)
plt.imsave("./myresult.png", disp_to_img, cmap='plasma')
#torch.save(net, "./model_city2kitti")

