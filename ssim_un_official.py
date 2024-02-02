import torch
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])

    return gauss / gauss.sum()


def create_window(window_size, channel,window_type = "gaussion"):
    if window_type=="gaussion":
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    if window_type=="average":
        _2D_window = torch.ones(size=(1,1,window_size,window_size))

    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # Ux^2
    mu1_sq = mu1.pow(2)
    # Uy^2
    mu2_sq = mu2.pow(2)
    # Ux*Uy
    mu1_mu2 = mu1 * mu2

    # 计算image1的方差Var(X)=E[X^2]-E[X]^2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    # 计算image2的方差Var(Y)=E[Y^2]-E[Y]^2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    # 计算协方差cov(X,Y)=E[XY]-E[X]E[Y]
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:

        return ssim_map.mean()

    else:

        return ssim_map.mean(1).mean(1).mean(1)
def guide_filter(I,p,window_size,eps=0.01):
    channel = I.shape[1]
    window = create_window(window_size=window_size,channel=I.shape[1],window_type="average")
    if I.is_cuda:
        window = window.cuda(I.get_device())
    pad = nn.ReflectionPad2d(window_size//2)

    mu_i = (F.conv2d(pad(I), window, padding=0, groups=channel)/(window_size*window_size))
    mu_p = (F.conv2d(pad(p), window, padding=0, groups=channel)/(window_size*window_size))
    mu_ip = (F.conv2d(pad(I*p), window, padding=0, groups=channel)/(window_size*window_size))
    cov_ip = mu_ip-mu_i*mu_p
    mu_ii = (F.conv2d(pad(I*I), window, padding=0, groups=channel)/(window_size*window_size))
    var_i = mu_ii-mu_i*mu_i
    a = (cov_ip/(var_i+eps))
    b = mu_p-a*mu_i
    mu_a = (F.conv2d(pad(a), window, padding=0, groups=channel)/(window_size*window_size))
    mu_b = (F.conv2d(pad(b), window, padding=0, groups=channel)/(window_size*window_size))
    GIF_V = mu_a*I+mu_b
    return GIF_V



def ssim_end2end(img1, img2, img_fused, window, window_size, channel, size_average=True):
    pad = nn.ReflectionPad2d(window_size // 2)

    #||| enchance img1 detail|||
    q1 = guide_filter(I=img1,p=img1,window_size=11,eps=0.01)
    img1_detail = (img1-q1)
    img1 = 5*img1_detail+q1


    # |||  get target image |||
    mu1 = F.conv2d(pad(img1), window, padding=0, groups=channel)
    mu2 = F.conv2d(pad(img2), window, padding=0, groups=channel)
    contrast_1 = torch.norm(img1 - (img1+img2)/2, p=2, dim=[2,3], keepdim=True)
    contrast_2 = torch.norm(img2 - (img1+img2)/2, p=2, dim=[2,3], keepdim=True)
    target_contrast = torch.max(contrast_1, contrast_2)
    S1 = (img1 - mu1) / contrast_1
    S2 = (img2 - mu2) / contrast_2
    weight1 = torch.norm(img1, p=1, dim=[2, 3], keepdim=True) / (
            torch.norm(img1, p=1, dim=[2, 3], keepdim=True) + torch.norm(img2, p=1, dim=[2, 3], keepdim=True))
    weight2 = 1 - weight1
    target_structure = (weight1 * S1 + weight2 * S2)  # [8,1,256,256]
    # target_illu = (weight1*mu1+weight2*mu2)
    target = target_contrast * target_structure




    # ||| end |||
    # ||| if target image W/O illu need change|||
    mu_fused = F.conv2d(pad(img_fused), window, padding=0, groups=channel)
    contrast_fused = torch.norm(img_fused - mu_fused, p=2, dim=[2, 3], keepdim=True)
    S_fused = (img_fused - mu_fused) / contrast_fused
    img_fused = contrast_fused * S_fused
    # ||| end |||
    # 计算target与img_fused的均值，以及均值的平方，均值相乘
    mean_target = F.conv2d(pad(target), window, padding=0, groups=channel)
    mean_target_sq = mean_target.pow(2)
    mean_fused = F.conv2d(pad(img_fused), window, padding=0, groups=channel)
    mean_fused_sq = mean_fused.pow(2)
    mu1_mu2 = mean_target * mean_fused
    # 计算image1的方差Var(X)=E[X^2]-E[X]^2
    sigma1_sq = F.conv2d(pad(target * target), window, padding=0, groups=channel) - mean_target_sq
    # 计算image2的方差Var(Y)=E[Y^2]-E[Y]^2
    sigma2_sq = F.conv2d(pad(img_fused * img_fused), window, padding=0, groups=channel) - mean_fused_sq
    # 计算协方差cov(X,Y)=E[XY]-E[X]E[Y]
    sigma12 = F.conv2d(pad(target * img_fused), window, padding=0, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    # 只计算了C*S之间的区别，因此不添加像素值
    # ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mean_target_sq + mean_fused_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_map = ((2 * sigma12 + C2)) / ((sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# def ssim_end2end(img1, img2, img_fused, window, window_size, channel, size_average = True):
#     pad = nn.ZeroPad2d(window_size // 2)
#     mu1 = F.conv2d(pad(img1), window, padding = 0, groups = channel)
#     mu2 = F.conv2d(pad(img2), window, padding = 0, groups = channel)
#     contrast_1 = torch.norm(img1 - mu1,p=2,dim=[2,3],keepdim=True)
#     contrast_2 = torch.norm(img2 - mu2, p=2, dim=[2, 3], keepdim=True)
#     mu_f = F.conv2d(pad(img_fused),window,padding= 0, groups= channel)
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     muf_sq = mu_f.pow(2)
#
#     # 计算image1的方差Var(X)=E[X^2]-E[X]^2
#     sigma1_sq = F.conv2d(pad(img1*img1), window, padding = 0, groups = channel) - mu1_sq
#     # 计算image2的方差Var(Y)=E[Y^2]-E[Y]^2
#     sigma2_sq = F.conv2d(pad(img_fused*img_fused), window, padding = 0, groups = channel) - muf_sq
#     # 计算协方差cov(X,Y)=E[XY]-E[X]E[Y]
#     sigma12 = F.conv2d(pad(img1*img_fused), window, padding = 0, groups = channel) - mu1*mu_f
#
#     C2 = 0.03**2
#     ssim_map_1f = ((2 * sigma12 + C2)) / ((sigma1_sq + sigma2_sq + C2))
#
#     # 计算image1的方差Var(X)=E[X^2]-E[X]^2
#     sigma1_sq = F.conv2d(pad(img2 * img2), window, padding=0, groups=channel) - mu2_sq
#     # 计算image2的方差Var(Y)=E[Y^2]-E[Y]^2
#     sigma2_sq = F.conv2d(pad(img_fused * img_fused), window, padding=0, groups=channel) - muf_sq
#     # 计算协方差cov(X,Y)=E[XY]-E[X]E[Y]
#     sigma12 = F.conv2d(pad(img2 * img_fused), window, padding=0, groups=channel) - mu2 * mu_f
#     ssim_map_2f = ((2 * sigma12 + C2)) / ((sigma1_sq + sigma2_sq + C2))
#     ssim_map = ssim_map_1f*(mu1>mu2) + ssim_map_2f*(mu2>=mu1)
#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):

        super(SSIM, self).__init__()

        self.window_size = window_size

        self.size_average = size_average

        self.channel = 1

        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():

            window = self.window

        else:

            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())

            window = window.type_as(img1)

            self.window = window

            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM_end2end(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):

        super(SSIM_end2end, self).__init__()

        self.window_size = window_size

        self.size_average = size_average

        self.channel = 1

        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, img_fused):

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():

            window = self.window

        else:

            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())

            window = window.type_as(img1)

            self.window = window

            self.channel = channel

        return ssim_end2end(img1, img2, img_fused, window, self.window_size, channel, self.size_average)


class TV_Loss(torch.nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IF):
        r = IA - IF
        h = r.shape[2]
        w = r.shape[3]
        tv1 = torch.pow((r[:, :, 1:, :] - r[:, :, :h - 1, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:] - r[:, :, :, :w - 1]), 2).mean()
        return tv1 + tv2


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()

    window = create_window(window_size, channel)

    if img1.is_cuda:

        window = window.cuda(img1.get_device())

    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

