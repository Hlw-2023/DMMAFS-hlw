# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms
import torchvision.transforms as transforms
from glob import glob
import os
from PIL import Image
import numpy as np
from torchvision import models
_tensor = transforms.ToTensor()
_pil_rgb = transforms.ToPILImage('RGB')
_pil_gray = transforms.ToPILImage()
device = 'cuda'


class Dataset(Data.Dataset):
    def __init__(self, root, resize=256, gray=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.resize = resize
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        if self.gray:
            img = img.convert('L')
        img = self.transform(img)
        return img

class My_sampler(Data.sampler.Sampler):
    def __init__(self, dataset,indices):
        self.length = int(len(dataset))
        self.indices = list(range(self.length))
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return self.length

class Dataset_SEQ_BP_t(Data.Dataset):
    def __init__(self, root, c_path, gray=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.files.sort()
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        root = np.load(self.files[index])
        seq_fea = root['res_features']
        coords = root['res_coords']
        label_bp = root['BP_label_vec']
        label_go = root['BP_label']
        real_go = []
        for i, val in enumerate(label_go):
            if val != '0':
                real_go.append(val)
        label_bp = label_bp.astype(np.float32)
        seq_fea = self.transform(seq_fea).float()
        v = self.files[index]
        v = v.split("/")[-1].split('res.np')[0]
        c_root = np.load('/Datasets/hlw/train_c_hmfr/' + v + '.npz')
        cmap = c_root['res_m']
        cmap = cmap.astype(np.float32)
        cmap = self.transform(cmap)
        go = label_go.tolist()

        return seq_fea.squeeze(dim=0), real_go, cmap, go


class Dataset_SEQ_MF_t(Data.Dataset):
    def __init__(self, root, c_path, gray=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.files.sort()
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        root = np.load(self.files[index])
        seq_fea = root['res_features']
        coords = root['res_coords']
        label_mf = root['MF_label_vec']
        label_go = root['MF_label']
        real_go = []
        for i, val in enumerate(label_go):
            if val != '0':
                real_go.append(val)
        label_mf = label_mf.astype(np.float32)
        seq_fea = self.transform(seq_fea).float()
        v = self.files[index]
        v = v.split("/")[-1].split('res.np')[0]
        c_root = np.load('/Datasets/hlw/train_c_hmfr/' + v + '.npz')
        cmap = c_root['res_m']
        cmap = cmap.astype(np.float32)
        cmap = self.transform(cmap)
        go = label_go.tolist()

        return seq_fea.squeeze(dim=0), real_go, cmap, go


class Dataset_SEQ_CC_t(Data.Dataset):
    def __init__(self, root, c_path, gray=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.files.sort()
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        root = np.load(self.files[index])
        seq_fea = root['res_features']
        coords = root['res_coords']
        label_cc = root['CC_label_vec']
        label_go = root['CC_label']
        real_go = []
        for i, val in enumerate(label_go):
            if val != '0':
                real_go.append(val)
        label_cc = label_cc.astype(np.float32)
        seq_fea = self.transform(seq_fea).float()
        v = self.files[index]
        v = v.split("/")[-1].split('res.np')[0]
        c_root = np.load('/Datasets/hlw/train_c_hmfr/' + v + '.npz')
        cmap = c_root['res_m']
        cmap = cmap.astype(np.float32)
        cmap = self.transform(cmap)
        go = label_go.tolist()

        return seq_fea.squeeze(dim=0), real_go, cmap, go




class Dataset_SEQ_BP(Data.Dataset):
    def __init__(self, root, c_path, gray=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.files.sort()
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        root = np.load(self.files[index])
        seq_fea = root['res_features']
        coords = root['res_coords']
        label_bp = root['BP_label_vec']
        label_go = root['BP_label']
        real_go = []
        for i, val in enumerate(label_go):
            if val != '0':
                real_go.append(val)
        label_bp = label_bp.astype(np.float32)
        seq_fea = self.transform(seq_fea).float()
        v = self.files[index]
        v = v.split("/")[-1].split('res.np')[0]
        c_root = np.load('/Datasets/hlw/train_c_hmfr/' + v + '.npz')
        cmap = c_root['res_m']
        cmap = cmap.astype(np.float32)
        cmap = self.transform(cmap)

        return seq_fea.squeeze(dim=0), real_go, cmap

class Dataset_SEQ_MF(Data.Dataset):
    def __init__(self, root, c_path, gray=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.files.sort()
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        root = np.load(self.files[index])
        seq_fea = root['res_features']
        coords = root['res_coords']
        label_mf = root['MF_label_vec']
        label_go = root['MF_label']
        real_go = []
        for i, val in enumerate(label_go):
            if val != '0':
                real_go.append(val)
        label_mf = label_mf.astype(np.float32)
        seq_fea = self.transform(seq_fea).float()
        v = self.files[index]
        v = v.split("/")[-1].split('res.np')[0]
        c_root = np.load('/Datasets/hlw/train_c_hmfr/' + v + '.npz')
        cmap = c_root['res_m']
        cmap = cmap.astype(np.float32)
        cmap = self.transform(cmap)

        return seq_fea.squeeze(dim=0), real_go, cmap

class Dataset_SEQ_CC(Data.Dataset):
    def __init__(self, root, c_path, gray=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.files.sort()
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        root = np.load(self.files[index])
        seq_fea = root['res_features']
        coords = root['res_coords']
        label_cc = root['CC_label_vec']
        label_go = root['CC_label']
        real_go = []
        for i, val in enumerate(label_go):
            if val != '0':
                real_go.append(val)
        label_cc = label_cc.astype(np.float32)
        seq_fea = self.transform(seq_fea).float()
        v = self.files[index]
        v = v.split("/")[-1].split('res.np')[0]
        c_root = np.load('/Datasets/hlw/train_c_hmfr/' + v + '.npz')
        cmap = c_root['res_m']
        cmap = cmap.astype(np.float32)
        cmap = self.transform(cmap)

        return seq_fea.squeeze(dim=0), real_go, cmap






def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def load_img(img_path, img_type='gray'):
    img = Image.open(img_path)
    if img_type=='gray':
        img = img.convert('L')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)
    return img

def nolinear_trans_patch(x):

    b = torch.rand(1)
    x = torch.log(1+x)*b.item()
    return x


def gradient(input,blur = True):
    conv_op = nn.Conv2d(1, 1, 3, bias=False,padding=1)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # kernel = np.array([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], dtype='float32')
    # kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype='float32')
    kernel = kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = (torch.from_numpy(kernel)).to(device).type(torch.float32)
    gaussion = GaussianBlurConv(channels=1)
    if blur:
        edge_detect = conv_op(gaussion(input))
    else:
        edge_detect = conv_op(input)
    return edge_detect
def gradient_sobel(input):
    conv_op = nn.Conv2d(1, 1, 3, bias=False,padding=1)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    kernel_T = kernel.T
    kernel = kernel.reshape((1, 1, 3, 3))
    kernel_T = kernel_T.reshape((1, 1, 3, 3))
    conv_op.weight.data = (torch.from_numpy(kernel)).to(device).type(torch.float32)
    edge_detect_vertical = conv_op(input)
    conv_op.weight.data = (torch.from_numpy(kernel_T)).to(device).type(torch.float32)
    edge_detect_horizontal = conv_op(input)
    edge_detect = (edge_detect_vertical+edge_detect_horizontal)
    return edge_detect
def gradient_Isotropic(input):
    conv_op = nn.Conv2d(1, 1, 3, bias=False,padding=1)
    kernel = np.array([[1, 0, -1], [2**0.5, 0, -2**0.5], [1, 0, -1]], dtype='float32')
    kernel_T = np.array([[-1, -2**0.5, -1], [0, 0, 0], [1, 2**0.5, 1]], dtype='float32')
    kernel = kernel.reshape((1, 1, 3, 3))
    kernel_T = kernel_T.reshape((1, 1, 3, 3))
    conv_op.weight.data = (torch.from_numpy(kernel)).to(device).type(torch.float32)
    edge_detect_vertical = conv_op(input)
    conv_op.weight.data = (torch.from_numpy(kernel_T)).to(device).type(torch.float32)
    edge_detect_horizontal = conv_op(input)
    edge_detect = (edge_detect_vertical+edge_detect_horizontal)
    return edge_detect


def hist_similar(x,y):
    t_min = torch.min(torch.cat((x, y), 1)).item()
    t_max = torch.max(torch.cat((x, y), 1)).item()
    return (torch.norm((torch.histc(x, 255, min=t_min, max=t_max)-torch.histc(y, 255, min=t_min, max=t_max)),2))/255


def fusion_exp( a, b):
    expa = torch.exp(a)
    expb = torch.exp(b)
    pa = expa / (expa + expb)
    pb = expb / (expa + expb)

    return pa * a + pb * b







class VGGLoss(nn.Module):
    def __init__(self,end=5):
        super(VGGLoss, self).__init__()
        # device = 'cuda:3'
        self.vgg = Vgg19(end=end).to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0][:end]
        self.e = end

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.151, 0.131, 0.120]).reshape((1, 3, 1, 1))),
                                       requires_grad=False).to(device)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.037, 0.034, 0.031]).reshape((1, 3, 1, 1))),
                                      requires_grad=False).to(device)

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, end = 5,requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.e = end

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out[:self.e]


def mulGANloss(input_, is_real):
    criterionGAN = torch.nn.MSELoss()

    if is_real:
        label = 1
    else:
        label = 0
    # 判断输入是否为list类型
    if isinstance(input_[0], list):
        loss = 0.0
        for i in input_:
            pred = i[-1]
            target = torch.Tensor(pred.size()).fill_(label).to(pred.device)
            loss += criterionGAN(pred, target)
        return loss
    else:
        target = torch.Tensor(input_[-1].size()).fill_(label).to(input_[-1].device)
        return criterionGAN(input_[-1], target)


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device).type(torch.float32)
        self.pad = nn.ReflectionPad2d(2)
        self.conv_op = nn.Conv2d(channels,channels,kernel_size=5,stride=1,padding=0,bias=False)
        self.conv_op.weight.data = self.weight
    def forward(self, x):

        return self.conv_op(self.pad(x))

class Gradient_L1(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(Gradient_L1, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)





def patchify(imgs, channel):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = 16
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], channel, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * channel ))
    return x

def unpatchify(x,in_chans):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = 16
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], in_chans, h * p, h * p))
    return imgs