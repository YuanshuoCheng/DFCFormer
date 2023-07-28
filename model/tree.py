import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model.parts import LinearProjection,window_partition,window_reverse,Mlp,DoubleConv,SingleConv,QuadraticMerge,QuadraticSplitting

#from parts import LinearProjection,window_partition,window_reverse,patchPartition,patchMerge,Mlp,DoubleConv,SingleConv

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, qkv_bias=True, qk_scale=None,mlp_ratio=0.25):
        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # define a parameter table of relative position bias

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias,query=True)
        self.proj = Mlp(in_features=dim,out_features=dim,hidden_features=int(dim*mlp_ratio))
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        q, k, v = self.qkv(x)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class InterDomainAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=2,
                 mlp_ratio=0.5, qkv_bias=True, qk_scale=None,norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=(win_size,win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,mlp_ratio=mlp_ratio)
        #self.out_proj = nn.Linear(dim,dim)#Mlp(in_features=dim,out_features=dim,hidden_features=int(dim*mlp_ratio))
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        shifted_x = x
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        attn_windows = self.attn(x_windows)  # nW*B, win_size*win_size, C
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        x = shortcut + x
        return x


class Node(nn.Module):
    def __init__(self,patch_size,in_channels,base_kernel_size=2,num_heads=4,ratio=0.5):
        super(Node, self).__init__()
        if patch_size > base_kernel_size:
            self.subnod = Node(patch_size // 2,in_channels,base_kernel_size,num_heads,ratio)

        self.wintransformer = InterDomainAttentionBlock(dim=in_channels, num_heads=num_heads,
                                                      win_size=base_kernel_size,mlp_ratio=ratio)
        self.base_kernel_size = base_kernel_size
        self.patch_size = patch_size

    def forward(self,x):
        B,C,H,W = x.shape
        if H == self.base_kernel_size:
            x = x.view(B,C,H*W).permute(0,2,1)
            x = self.wintransformer(x)
            res = x.permute(0,2,1).view(B,C,H,W)
            return res
        else:
            maps = QuadraticSplitting(x)
            B_, N_, C_, H_, W_ = maps.shape
            patches = maps.reshape(B_*N_,C_,H_,W_)
            patches = self.subnod(patches)
            patches = patches.reshape(B_,N_,C_,H_,W_)
            x = QuadraticMerge(patches)
            x = x.view(B, C, H * W).permute(0, 2, 1)
            x = self.wintransformer(x)
            x = x.permute(0, 2, 1).view(B, C, H, W)
            return x


class GAT(nn.Module):
    def __init__(self,patch_size,in_channels,out_channels,base_kernel_size=2,num_heads=8,ratio=0.5):
        super(GAT, self).__init__()
        self.in_proj = SingleConv(in_channels,in_channels//4)
        self.tree = Node(patch_size,in_channels//4,base_kernel_size=base_kernel_size,num_heads=num_heads,ratio=ratio)
        self.out_proj = nn.Conv2d(in_channels//4,out_channels,1,1)
    def forward(self,x):
        x = self.in_proj(x)
        x = self.tree(x)
        x = self.out_proj(x)
        return x

if __name__ == '__main__':
    x = torch.randn(size=(1,64,128,128)).cuda()
    model = GAT(patch_size=128,in_channels=64,out_channels=64).cuda()
    res = model(x)
    print(res.shape)





