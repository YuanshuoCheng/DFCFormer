import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows
class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, bias=True,query=False,ratio=1):
        super().__init__()
        self.ratio = ratio
        inner_dim = dim_head * heads
        self.heads = heads
        self.query = query
        if query:
            self.to_q = nn.Linear(dim, int(inner_dim*self.ratio), bias=bias)
        self.to_kv = nn.Linear(dim, int(inner_dim*(1+self.ratio)), bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x,global_tokens=None):
        B_, N, C = x.shape
        attn_kv = x
        kv_N = N
        if global_tokens is not None:
            attn_kv = torch.cat([attn_kv,global_tokens],dim=1)
            kv_N = kv_N+global_tokens.shape[1]
        kv = self.to_kv(attn_kv).reshape(B_, kv_N, self.heads, int(C // self.heads*(1+self.ratio))).permute(0,2,1,3)
        k, v = kv[:,:,:,:int(C//self.heads*self.ratio)], kv[:,:,:,int(C//self.heads*self.ratio):] #[bs,n,heads,c//headers*1.5]->,bs,heads,n,c//headers*1.5]
        if self.query:
            q = self.to_q(x).reshape(B_, N, 1, self.heads, int(C // self.heads*self.ratio)).permute(2, 0, 3, 1, 4)
            q = q[0]
            return (q,k,v)
        return (k, v)

    def flops(self, H, W):
        flops = H * W * self.dim * self.inner_dim * 3
        return flops



def QuadraticSplitting(x):
    B,C,H,W = x.shape
    res = F.unfold(x,kernel_size=H//2,stride=1,padding=0,dilation=2)
    res = res.permute(0, 2, 1).reshape(B, 4, C, H//2, W//2)
    return res

def QuadraticMerge(x):
    B,N,C,H,W = x.shape
    res = x.reshape(B,N,C*H*W).permute(0,2,1)
    res = F.fold(res, kernel_size=H, stride=1, padding=0, dilation=2, output_size=(H*2, W*2))
    return res

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels,stride=1,kernel_size=3,norm=nn.BatchNorm2d):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2),
            norm(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2,stride=stride),
            norm(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_conv(x)
class MobileConv(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1,padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                      padding=padding,groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False),  # pading=0
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride,
                      padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),  # pading=0
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1,norm=nn.BatchNorm2d):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            norm(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_conv(x)
class Head(nn.Module):
    def __init__(self, in_channels, out_channels,mid_channels,norm=nn.BatchNorm2d):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels,mid_channels,kernel_size=1,stride=1)
        self.conv1 = DoubleConv(mid_channels,mid_channels,mid_channels)
        self.conv2 = DoubleConv(mid_channels,mid_channels,mid_channels)
        self.out = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2),
            norm(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x = self.in_proj(x)
        x = self.conv1(x)+x
        x = self.conv2(x)+x
        return self.out(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpBlock,self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = MobileConv(in_channels, out_channels)


    def forward(self, x1, x2):  # 拼接，其中x2的大小大于x1
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])  # 左右上下
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(DownBlock,self).__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            MobileConv(in_channels=in_channels, out_channels=out_channels,stride=2,padding=1)
        )

    def forward(self, x):
        res = self.maxpool_conv(x)
        return res,x



def subdomainSplit(x,n_subdomain):
    bs,c,h,w = x.shape
    h_split_size = h//n_subdomain
    h_remainder = h%n_subdomain
    w_split_size = w // n_subdomain
    w_remainder = w % n_subdomain

    lt_s = [h_split_size+1,w_split_size+1]
    rb_s = [h_split_size,w_split_size]
    if h_remainder==0:
        lt_s[0] = h_split_size
        h_remainder = n_subdomain//2
    if w_remainder==0:
        lt_s[1] = w_split_size
        w_remainder = n_subdomain//2

    lt_data = (x[:,:,:lt_s[0]*h_remainder,:lt_s[1]*w_remainder],lt_s[0],lt_s[1],
               h_remainder,w_remainder)
    rt_data = (x[:,:,:lt_s[0]*h_remainder,lt_s[1]*w_remainder:],lt_s[0],rb_s[1],
               h_remainder,n_subdomain-w_remainder)
    lb_data = (x[:,:,lt_s[0]*h_remainder:,:lt_s[1]*w_remainder],rb_s[0],lt_s[1],
               n_subdomain-h_remainder,w_remainder)
    rb_data = (x[:,:,lt_s[0]*h_remainder:,lt_s[1]*w_remainder:],rb_s[0],rb_s[1],
               n_subdomain-h_remainder,n_subdomain-w_remainder)
    return [lt_data,rt_data,lb_data,rb_data]


def subdomainPartition(x,kernel_size,size):
    B,C,H,W = x.shape
    res = F.unfold(x,kernel_size=kernel_size,stride=kernel_size)
    res = res.permute(0, 2, 1).view(B, size[0]*size[1], C, kernel_size[0], kernel_size[1]).contiguous()
    return res

def subdomainMerge(x,kernel_size,size):
    B,N,C,H,W = x.shape
    res = x.reshape(B,N,C*H*W).permute(0,2,1).contiguous()
    res = F.fold(res, kernel_size=kernel_size, stride=kernel_size,output_size=(size[0], size[1]))
    return res


