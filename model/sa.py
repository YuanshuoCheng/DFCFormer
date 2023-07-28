import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import math
import numpy as np
import time
from torch import einsum
# from tree import InterDomainAttention
# from parts import LinearProjection,DoubleConv,subdomainSplit,subdomainMerge,subdomainPartition,Mlp,SingleConv

from model.tree import GAT
from model.parts import LinearProjection,DoubleConv,subdomainSplit,subdomainMerge,subdomainPartition,Mlp,SingleConv

class IntraDomainAttention(nn.Module):
    def __init__(self, dim,num_heads=8,ratio=0.5):
        super().__init__()
        self.ratio=ratio
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads,query=True,ratio=self.ratio)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x,global_tokens): #[bs,n,c]
        B_, N, C = x.shape
        q,k,v = self.qkv(x,global_tokens)#[bs,heads,n,c//headers]
        attn = (q @ k.transpose(-2, -1))
        attn = attn #+ relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_,N, C)
        return x


class DomainEmbeding(nn.Module):
    def __init__(self, dim, query_size=4,num_heads=8,ratio=0.25):
        super().__init__()
        self.ratio=ratio
        self.dim = dim
        self.num_heads = num_heads
        self.kv = LinearProjection(dim, num_heads, dim // num_heads,self.ratio)
        self.softmax = nn.Softmax(dim=-1)
        self.query_size = query_size
        self.domain_query = nn.Parameter(torch.randn(size=(num_heads,query_size,dim//num_heads)))
    def forward(self, x):
        q = self.domain_query
        B_, N, C = x.shape
        k, v = self.kv(x)
        attn = (q @ k.transpose(-2, -1))
        attn = attn #+ relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_,self.query_size, self.dim)
        return x


class GABBlock(nn.Module):
    def __init__(self,in_channels=64,query_size=4,num_heads = 4,ratio=0.5,n_subdomains=128):
        super().__init__()
        self.num_heads = num_heads
        self.n_subdomain = n_subdomains
        self.query_size = query_size
        #self.conv = DoubleConv(in_channels,in_channels,in_channels//2)
        self.conv = SingleConv(in_channels=in_channels, out_channels=in_channels)
        self.domain_embeding = DomainEmbeding(dim=in_channels,query_size=self.query_size,num_heads=num_heads,ratio=ratio)
        self.intra_domain_attention = IntraDomainAttention(dim=in_channels,num_heads=num_heads,ratio=ratio)
        self.gat = GAT(patch_size=self.n_subdomain,in_channels=in_channels*4,out_channels=in_channels*4,
                                                           ratio=ratio,num_heads=num_heads)
        self.out_proj = nn.Conv2d(in_channels,in_channels,3,1,padding=1)

    def forward(self, x):
        x_ori = x
        x = subdomainSplit(self.conv(x),self.n_subdomain)
        x1 = []
        subdomains = []
        for region in x:
            subdomain_data,subdomain_h,subdomain_w,n_subdomain_h,n_subdomain_w = region
            #print(subdomain_data.shape) # 分成左上左下右上右下四个区域
            data = subdomainPartition(subdomain_data,kernel_size=(subdomain_h,subdomain_w),
                                       size=(n_subdomain_h,n_subdomain_w))#[bs,n,c,h,w]
            x1.append((data,subdomain_h,subdomain_w,n_subdomain_h,n_subdomain_w))
            #print(data.shape) #每个区域切patch，方便并行计算torch.Size([1, 4096, 32, 2, 2])
            bs,n,c,h,w = data.shape
            #data = data.view(-1,c,h,w)
            data = data.view(bs*n,c,h*w).permute(0, 2, 1)#[bs*n,h*w,c]
            data = self.domain_embeding(data).permute(0, 2, 1) # B_HW, self.dim,self.query_size
            #data = data.reshape(bs,n_subdomain_h,n_subdomain_w,c,self.query_size)#.permute(0,3,1,2)
            subdomains.append(data.reshape(bs,n_subdomain_h,n_subdomain_w,c,self.query_size))
        top = torch.cat([subdomains[0],subdomains[1]],dim=2)
        btm = torch.cat([subdomains[2],subdomains[3]],dim=2)
        x = torch.cat([top,btm],dim=1) #([1, 128, 128, 32, 4])
        b,h,w,c,n_q = x.shape
        x = x.permute(0,4,3,2,1)
        inter_att = self.gat(x.reshape(b,n_q*c,h,w))
        inter_att = inter_att.view(bs,c,self.query_size,self.n_subdomain,self.n_subdomain)
        # torch.Size([1, 8, 4, 128, 128])
        _, _, _, lt_n_subdomain_h, lt_n_subdomain_w = x1[0]
        res = []
        for i,region in enumerate(x1):
            subdomain_data, subdomain_h, subdomain_w, n_subdomain_h, n_subdomain_w = region
            if i == 0:
                global_tokens = inter_att[:,:,:,:lt_n_subdomain_h,:lt_n_subdomain_w]
            elif i==1:
                global_tokens = inter_att[:,:,:,:lt_n_subdomain_h,lt_n_subdomain_w:]
            elif i==2:
                global_tokens = inter_att[:,:,:,lt_n_subdomain_h:,:lt_n_subdomain_w]
            elif i==3:
                global_tokens = inter_att[:,:,:,lt_n_subdomain_h:,lt_n_subdomain_w:] #[bs,c,16,h,w]
            b,c,n_q,h,w = global_tokens.shape
            global_tokens = global_tokens.permute(0,3,4,2,1).reshape(b*h*w,n_q,c)#[b,h,w,nq,c]->[b*h*w,n_q,c]
            B,N,C,H,W = subdomain_data.shape
            subdomain_data = subdomain_data.view(B*N,C,H*W).permute(0,2,1)#[B*N,H*W,C]
            subdomain_data = self.intra_domain_attention(subdomain_data,global_tokens)#[B*N,H*W,C],[b*h*w,n_q,c]

            subdomain_data = subdomain_data.view(B,N,H,W,C).permute(0,1,4,2,3) #[b,n,c,h,w]
            subdomain_data = subdomainMerge(subdomain_data,kernel_size=(subdomain_h,subdomain_w),
                                size=(n_subdomain_h*subdomain_h,n_subdomain_w*subdomain_w))#[B,C,H,W]
            res.append(subdomain_data)
        top = torch.cat([res[0],res[1]],dim=3)
        btm = torch.cat([res[2],res[3]],dim=3)
        res = torch.cat([top,btm],dim=2)
        res = self.out_proj(res)+x_ori
        return res



