from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
from einops import rearrange
import pdb
from utils import patchify,unpatchify
import time
import torchvision
import os

import seaborn as sns
import matplotlib.pyplot as plt




class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = self.To_2tuple(img_size)
        patch_size = self.To_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def To_2tuple(self,x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class SequenceEmbed(nn.Module):
    """
    Sequence to Embedding
    """
    def __init__(self):
        super(SequenceEmbed, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(10, 20, 3, 2)
        self.max_pool2 = nn.MaxPool1d(3, 2)
        self.conv3 = nn.Conv1d(20, 40, 3, 2)
        self.linear1 = nn.Linear(40*14, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 40*14)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads=10, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_block_2d(

        x, drop_prob: float = 0.1, block_size: int = 7,  gamma_scale: float = 1.0,
        with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
        (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(torch.arange(W).to(x.device), torch.arange(H).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster

        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(
        x: torch.Tensor, drop_prob: float = 0.1, block_size: int = 7,
        gamma_scale: float = 1.0, with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    if batchwise:
        # one mask for whole batch, quite a bit faster
        block_mask = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device) < gamma
    else:
        # mask per batch element
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """
    def __init__(self,
                 drop_prob=0.1,
                 block_size=7,
                 gamma_scale=1.0,
                 with_noise=False,
                 inplace=False,
                 batchwise=False,
                 fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)
        else:
            return drop_block_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

#     输入两个模态的tensor，[x，y]，最终x的v作为输出
class Attention_within_2_modal(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        Bx, Nx, Cx = x.shape
        By,Ny,Cy = y.shape
        qkv_x = self.qkv_x(x).reshape(Bx, Nx, 3, self.num_heads, Cx // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_y = self.qkv_y(y).reshape(By, Ny, 3, self.num_heads, Cy // self.num_heads).permute(2, 0, 3, 1, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]   # make torchscript happy (cannot use tensor as tuple)
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
        attn = (qx @ ky.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y = (attn @ vy).transpose(1, 2).reshape(Bx, Nx, Cx)
        x = self.proj(y) + x
        x = self.proj_drop(x)
        return y


class cross_attn(nn.Module):
    def __init__(self,encoder_embed_dim=512,num_heads=16,attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.cmap_squeeze = nn.Linear(1000,encoder_embed_dim)
        self.qkv_cmap = nn.Linear(encoder_embed_dim,encoder_embed_dim*3)
        self.qkv_seq = nn.Linear(encoder_embed_dim,encoder_embed_dim*3)

        self.num_heads = num_heads
        head_dim = encoder_embed_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = Mlp(encoder_embed_dim*2,encoder_embed_dim*4,encoder_embed_dim)

    def forward(self,seq,cmap):
        cmap_qkv = self.cmap_squeeze(cmap)
        B, N, C = cmap_qkv.shape
        cmap_qkv =  self.qkv_cmap(cmap_qkv).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_c, k_c, v_c = cmap_qkv[0], cmap_qkv[1], cmap_qkv[2]
        seq_qkv = self.qkv_seq(seq).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_s, k_s, v_s = seq_qkv[0],seq_qkv[1],seq_qkv[2]


        attn = (q_s @ k_c.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        seq_attn = (attn @ v_s).transpose(1, 2).reshape(B, N, C)
        # v_ss = rearrange(v_s, 'f h w c -> f w (h c)')
        # v_cc = rearrange(v_c, 'f h w c -> f w (h c)')
        #
        # s_c = torch.cat([v_cc, v_ss], dim=2)
        # s_c = self.proj(s_c)


        seq_fuse = torch.cat([seq,seq_attn],dim=2)      # seq_fuse (1,1000,1024)
        # seq_fuse = torch.cat([seq, s_c], dim=2)
        seq = self.proj(seq_fuse)   # seq(1,1000,512)
        seq = self.proj_drop(seq)
        return seq  # seq(1,1000,512)

class cross_attn_xr(nn.Module):
    def __init__(self,encoder_embed_dim=512,num_heads=16,attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.cmap_squeeze = nn.Linear(1000,encoder_embed_dim)
        self.qkv_cmap = nn.Linear(encoder_embed_dim,encoder_embed_dim*3)
        self.qkv_seq = nn.Linear(encoder_embed_dim,encoder_embed_dim*3)

        self.num_heads = num_heads
        head_dim = encoder_embed_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = Mlp(encoder_embed_dim*2,encoder_embed_dim*4,encoder_embed_dim)

    def forward(self,seq,cmap):
        cmap_qkv = self.cmap_squeeze(cmap)
        B, N, C = cmap_qkv.shape
        cmap_qkv =  self.qkv_cmap(cmap_qkv).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_c, k_c, v_c = cmap_qkv[0], cmap_qkv[1], cmap_qkv[2]
        seq_qkv = self.qkv_seq(seq).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_s, k_s, v_s = seq_qkv[0],seq_qkv[1],seq_qkv[2]


        attn = (q_s @ k_c.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # seq_attn = (attn @ v_s).transpose(1, 2).reshape(B, N, C)
        v_ss = rearrange(v_s, 'f h w c -> f w (h c)')
        v_cc = rearrange(v_c, 'f h w c -> f w (h c)')

        s_c = torch.cat([v_cc, v_ss], dim=2)
        s_c = self.proj(s_c)


        # seq_fuse = torch.cat([seq,seq_attn],dim=2)      # seq_fuse (1,1000,1024)
        seq_fuse = torch.cat([seq, s_c], dim=2)
        seq = self.proj(seq_fuse)   # seq(1,1000,512)
        seq = self.proj_drop(seq)
        return seq  # seq(1,1000,512)

def stage_layer(embed_dim,num_heads,mlp_ratio,norm_layer,step_depth,):
    layer = [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                        for i in range(step_depth)]
    return nn.Sequential(*layer)

if __name__=="__main__":
    trans = EWS().cuda()
    tensor_x = torch.randn(size=[12, 138, 512]).cuda()
    tensor_y = torch.randn(size=[12, 138, 512]).cuda()
    tensor_x = trans(tensor_x,tensor_y)
    print(tensor_x.shape)

