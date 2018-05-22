
# coding: utf-8

# In[14]:


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
from MonodepthModel_monocular import *
import scipy.misc
from IPython.display import clear_output
from flow_warp.Optical_Flow_Warping_Tensorflow.flownet2_pytorch.networks.submodules import *
from flow_warp.Optical_Flow_Warping_Tensorflow.flownet2_pytorch.networks.resample2d_package.modules.resample2d import Resample2d
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

#torch.cuda.set_device(1)
os.environ[ " CUDA_VISIBLE_DEVICES "] = " 2 "


# In[15]:


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


# In[16]:


#args = get_args()
net = MonodepthNet().cuda()
#net = torch.load("/eva_data/hylai_model/monocular_model/model_epoch40")
#net = torch.load("/home/hylai/monodepth/model_city2kitti")
#optimizer = optim.Adam(net.parameters(), lr=1e-4)
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
params = list(net.parameters())
name = list(net.named_parameters())


# In[17]:


def get_data(path = '/eva_data/hylai_model/my/output/folder/'):
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


# In[18]:


def get_flow_data(path = '/eva_data/hylai_model/dataset/kitti/'):
    file_path_train = '/home/hylai/monodepth/utils/filenames/kitti_flow_train_files.txt'
    file_path_test = '/home/hylai/monodepth/utils/filenames/kitti_flow_test_files.txt'
    f_train = open(file_path_train)
    f_test = open(file_path_test)
    former_image_train = list()
    latter_image_train = list()
    flow_train = list()
    former_image_test = list()
    latter_image_test = list()
    
    num_train = 0
    num_test = 0
    
    for line in f_train:
        num_train += 1
        former_image_train.append(path+line.split()[0])
        latter_image_train.append(path+line.split()[1])
        flow_train.append(path+line.split()[2])
    for line in f_test:
        num_test += 1
        former_image_test.append(path+line.split()[0])
        latter_image_test.append(path+line.split()[1])
        
    return former_image_train, latter_image_train, flow_train, former_image_test, latter_image_test
a, b, c, d, e = get_flow_data()
#former = scipy.misc.imread(a[0], mode="RGB")
#latter = scipy.misc.imread(b[0], mode="RGB")
former = Image.open(a[0]).convert('RGB')
latter = Image.open(b[0]).convert('RGB')
former = former.transpose(Image.FLIP_LEFT_RIGHT)
latter = latter.transpose(Image.FLIP_LEFT_RIGHT)
#f = scipy.misc.imread(c[0], mode="RGB")
f = Image.open(c[0]).convert('RGB')
f = f.transpose(Image.FLIP_LEFT_RIGHT)
image = transforms.ToTensor()(latter).unsqueeze(0)
print(image.shape)
flow = transforms.ToTensor()(f).unsqueeze(0)
flow[:,0,:,:] = - (flow[:,0,:,:] - 0.5) * flow.shape[3]
flow[:,1,:,:] = (flow[:,1,:,:] - 0.5) * flow.shape[2]
flow = flow[:,:2,:,:]
print(flow.max(), flow.min())
x = Resample2d()(Variable(image.cuda()), Variable(flow.cuda()))
out = x.squeeze(0).data.cpu().numpy().transpose((1,2,0))
plt.imshow(out)


# In[19]:


def get_transform():
    return transforms.Compose([
        transforms.Scale([512, 256]),
        transforms.ToTensor()
    ])


# In[20]:


class myImageFolder(data.Dataset):
    def __init__(self, left, right, training, flow = None):
        self.right = right
        self.left = left
        self.training = training
        self.flow = flow
        
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        left_image = Image.open(left).convert('RGB')
        right_image = Image.open(right).convert('RGB')
        
        if self.flow is not None:
            flow = self.flow[index]
            flow_image = Image.open(flow).convert('RGB')
        
        #augmentation
        if self.training:
            
            #randomly flip
            if random.uniform(0, 1) > 0.5:
                left_image = left_image.transpose(Image.FLIP_LEFT_RIGHT)
                right_image = right_image.transpose(Image.FLIP_LEFT_RIGHT)
                if self.flow is not None:
                    flow_image = flow_image.transpose(Image.FLIP_LEFT_RIGHT)
                
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
        
        if self.flow is not None:
            flow_image = process(flow_image)
            flow_image[0,:,:] = - (flow_image[0,:,:] - 0.5) * flow_image.shape[2]
            flow_image[1,:,:] = (flow_image[1,:,:] - 0.5) * flow_image.shape[1]
            return left_image, right_image, flow_image
        
        return left_image, right_image
    def __len__(self):
        return len(self.left)


# In[21]:


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


# In[22]:


def gradient_x(img):
    gx = torch.add(img[:,:,:-1,:], -1, img[:,:,1:,:])
    return gx

def gradient_y(img):
    gy = torch.add(img[:,:,:,:-1], -1, img[:,:,:,1:])
    return gy

def get_disparity_smoothness(disp, pyramid):
    #filters = Variable(torch.cuda.FloatTensor([1,0]).view(1,2,1,1))
    #disp = [F.conv2d(disp[i], filters) for i in range(4)]
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]
    
    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]
    
    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
    
    return smoothness_x + smoothness_y


# In[23]:


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)
    
    #(input, kernel, stride, padding)
    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y
    
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    SSIM = SSIM_n / SSIM_d
    
    return torch.clamp((1 - SSIM) / 2, 0, 1)


# In[24]:


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


# In[25]:


left_image_train, right_image_train, left_image_test, right_image_test, num_train, num_test = get_data()
TrainImageLoader = torch.utils.data.DataLoader(
         myImageFolder(left_image_train, right_image_train, True), 
         batch_size = 8, shuffle = True, num_workers = 8, drop_last =False)
TestImageLoader = torch.utils.data.DataLoader(
         myImageFolder(left_image_test, right_image_test, False),
         batch_size = 8, shuffle = False, num_workers = 8, drop_last =False)

former_train, latter_train, flow_train, former_test, latter_test = get_flow_data()
TrainFlowLoader = torch.utils.data.DataLoader(
         myImageFolder(former_train, latter_train, True, flow_train), 
         batch_size = 8, shuffle = True, num_workers = 8, drop_last =False)
TestFlowLoader = torch.utils.data.DataLoader(
         myImageFolder(former_test, latter_test, False),
         batch_size = 8, shuffle = False, num_workers = 8, drop_last =False)

#Train
do_stereo = 0
alpha_image_loss = 0.85
disp_gradient_loss_weight = 0.1
lr_loss_weight = 1.0
num_epochs = 10
optimizer = optim.Adam(net.parameters(), lr = 0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32((3/5) * num_epochs), np.int32((4/5) * num_epochs)], gamma=0.5)


# In[ ]:


for epoch in range(1, 51):
    for batch_idx, (left, right) in enumerate(TrainImageLoader, 0):
        print(epoch, batch_idx)

        optimizer.zero_grad()
        #generate image pyramid[scale][batch]
        left_pyramid = make_pyramid(left, 4)
        right_pyramid = make_pyramid(right, 4)
        
        if do_stereo:
            model_input = Variable(torch.cat((left, right), 1).cuda())
        else:
            model_input = Variable(left.cuda())
        
        disp_est_left, disp_est_right = net(model_input)
        """
        tmp = []
        tmp2 = []
        for i in range(4):
            tmp.append(Variable(torch.zeros(disp_est_left[i].shape[0], 1, disp_est_left[i].shape[2], disp_est_left[i].shape[3]).cuda()))
            tmp2.append(Variable(torch.zeros(disp_est_left[i].shape[0], 1, disp_est_left[i].shape[2], disp_est_left[i].shape[3]).cuda()))
        disp_est_left_tmp = [torch.cat((disp_est_left[i], tmp[i]), 1) * disp_est_left[i].shape[3] for i in range(4)]
        disp_est_right_tmp = [torch.cat((disp_est_right[i], tmp2[i]), 1) * disp_est_right[i].shape[3] for i in range(4)]
        """
        filters = Variable(torch.cuda.FloatTensor([1,0]).view(2,1,1,1))
        filters2 = Variable(torch.cuda.FloatTensor([1,0]).view(2,1,1,1))
        disp_est_left_tmp = [F.conv2d(disp_est_left[i], filters) * disp_est_left[i].shape[3] for i in range(4)]
        disp_est_right_tmp = [F.conv2d(disp_est_right[i], filters2) * disp_est_right[i].shape[3] for i in range(4)]
        
        left_est = [Resample2d()(right_pyramid[i], -disp_est_left_tmp[i]) for i in range(4)]
        disp_left_smoothness = get_disparity_smoothness(disp_est_left, left_pyramid)
        l1_left = [torch.abs(left_est[i] - left_pyramid[i]) for i in range(4)]
        l1_reconstruction_loss_left = [torch.mean(l) for l in l1_left]
        ssim_left = [SSIM(left_est[i], left_pyramid[i]) for i in range(4)]
        ssim_loss_left = [torch.mean(s) for s in ssim_left]
        image_loss_left  = [alpha_image_loss * ssim_loss_left[i] + (1 - alpha_image_loss) * l1_reconstruction_loss_left[i] for i in range(4)]
        image_loss = sum(image_loss_left)
        disp_left_loss  = [torch.mean(torch.abs(disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
        disp_gradient_loss = sum(disp_left_loss)
        
        total_loss = image_loss + disp_gradient_loss_weight * disp_gradient_loss
        
        right_est = [Resample2d()(left_pyramid[i], disp_est_right_tmp[i]) for i in range(4)]
        disp_right_smoothness = get_disparity_smoothness(disp_est_right, right_pyramid)
        l1_right = [torch.abs(right_est[i] - right_pyramid[i]) for i in range(4)]
        l1_reconstruction_loss_right = [torch.mean(l) for l in l1_right]
        ssim_right = [SSIM(right_est[i], right_pyramid[i]) for i in range(4)]
        ssim_loss_right = [torch.mean(s) for s in ssim_right]
        image_loss_right  = [alpha_image_loss * ssim_loss_right[i] + (1 - alpha_image_loss) * l1_reconstruction_loss_right[i] for i in range(4)]
        image_loss_2 = sum(image_loss_right)
        disp_right_loss  = [torch.mean(torch.abs(disp_right_smoothness[i]))  / 2 ** i for i in range(4)]
        disp_gradient_loss_2 = sum(disp_right_loss)
        

        #LR consistency
        right_to_left_disp = [Resample2d()(disp_est_right[i], -disp_est_left_tmp[i]) for i in range(4)]
        left_to_right_disp = [Resample2d()(disp_est_left[i], disp_est_right_tmp[i]) for i in range(4)]
        
        #LR consistency
        scale = [disp_est_left[i].shape[3] for i in range(4)]
        lr_left_loss  = [torch.mean(torch.abs(right_to_left_disp[i] - disp_est_left[i])) for i in range(4)]
        lr_right_loss = [torch.mean(torch.abs(left_to_right_disp[i] - disp_est_right[i])) for i in range(4)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        
        #Total loss
        total_loss_2 = image_loss_2 + disp_gradient_loss_weight * disp_gradient_loss_2
        loss = total_loss + total_loss_2 + lr_loss_weight * lr_loss
        print(total_loss, total_loss_2)
        loss.backward()
        print(net.conv1.weight.grad[0,0,0,0])
        optimizer.step()
        
        if batch_idx % 100 == 0:
            clear_output()
    if epoch % 5 == 0:
        torch.save(net, "/eva_data/hylai_model/monocular_model/model_epoch" + str(epoch))


# In[16]:


for batch_idx, (left, right) in enumerate(TestImageLoader, 0):
    #model_input = Variable(torch.cat((right, left), 1).cuda())
    model_input = Variable(left.cuda())
    disp_est_left, disp_est_right = net(model_input)
    print(disp_est_left[0][0,0,:,:], disp_est_left[0].shape)
    plt.imshow(disp_est_left[0][0,0,:,:].data.cpu().numpy())
    disp_pp = post_process_disparity(disp_est_left[0][0,:,:,:].data.cpu().numpy())
    disp_to_img = scipy.misc.imresize(disp_pp, [375, 1242])
    #print(disp_to_img)
    #plt.imshow(disp_to_img)
    break


# In[ ]:


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
print(disp_to_img)

