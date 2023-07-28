import torch
import torch.nn as nn
from model.sa import GABBlock
from model.parts import DoubleConv,SingleConv,Head
from model.ca import MultiScaleChannelTransformerBlock,CTCA,Attention1x1,LayerNorm
from model.unet import UNET
import numpy as np


class GAB(nn.Module):
    def __init__(self,in_channels=64,scales=[1,2,4,8],n_subdomains=64):
        super().__init__()
        self.sa = GABBlock(in_channels=in_channels, query_size=4, ratio=0.5, n_subdomains=n_subdomains)
        self.ca = MultiScaleChannelTransformerBlock(dim=in_channels, ffn_expansion_factor=0.5, bias=False,
                                              LayerNorm_type='WithBias',scales=scales)
        self.out = nn.Sequential(SingleConv(in_channels=in_channels*2,out_channels=in_channels),
                                 nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1))
    def forward(self,x):
        x1 = self.sa(x)
        x2 = self.ca(x)
        return self.out(torch.cat([x1,x2],dim=1))+x

class FCB(nn.Module):
    def __init__(self,in_channels,query_size=64,hw=128):
        super().__init__()
        self.norm1 = LayerNorm(in_channels, 'BiasFree')
        self.crossatt = CTCA(in_channels,query_size,hw=hw)
        self.norm2 = LayerNorm(query_size, 'BiasFree')
        self.selfatt = Attention1x1(query_size)
        self.norm3 = LayerNorm(query_size, 'BiasFree')
        self.ffn = nn.Sequential(
            nn.Conv2d(query_size,query_size,1,1),
            nn.ReLU(),
            nn.Conv2d(query_size, query_size, 1, 1)
        )
    def forward(self,x,feat_q):
        feat_q_1 = self.crossatt(self.norm1(x),feat_q)+feat_q
        feat_q_2 = feat_q_1+self.selfatt(self.norm2(feat_q_1))
        feat_q_3 = feat_q_2+self.ffn(self.norm3(feat_q_2))
        return feat_q_3




class DFCFormer(nn.Module):
    def __init__(self,in_channels = 3,out_channels=3,base_channels=64,query_size=32,query_hw=32):
        super().__init__()
        self.base_channels = base_channels
        self.query_size = query_size
        self.feat_q = nn.Parameter(torch.randn((1,query_size,query_hw,query_hw)))
        self.in_proj = DoubleConv(in_channels=in_channels,out_channels=base_channels,mid_channels=16,kernel_size=3)
        self.block1 = GAB(base_channels,scales=[4]*4,n_subdomains=32)
        self.cluster1 = FCB(base_channels,query_size=query_size,hw=query_hw)
        self.block2 = GAB(base_channels,scales=[2]*4,n_subdomains=64)
        self.cluster2 = FCB(base_channels,query_size=query_size, hw=query_hw)
        self.block3 = GAB(base_channels,scales=[1]*4,n_subdomains=128)
        self.cluster3 = FCB(base_channels,query_size=query_size, hw=query_hw)
        # self.block4 = DABlock(base_channels, scales=[1] * 4, n_subdomains=128)
        # self.cluster4 = FCB(base_channels,query_size=query_size, hw=query_hw)
        self.out_proj = nn.Conv2d(in_channels=base_channels,out_channels=base_channels,kernel_size=1,stride=1)
        self.cluster_out = CTCA(base_channels,query_size,hw=query_hw,pool_V=False)

        #self.instance_norm = nn.InstanceNorm2d(query_size)
        self.noise_proj = nn.Conv2d(in_channels=self.query_size//4,out_channels=self.query_size//4*3,stride=1,kernel_size=1)
        #self.noise_head = Head(in_channels=self.query_size,out_channels=out_channels,mid_channels=64)
        self.clear_head = UNET(in_channels=self.query_size//4*3,out_channels=out_channels)

    def forward(self,x):
        feat_q = self.feat_q
        x = self.in_proj(x)
        x = self.block1(x)
        feat_q = self.cluster1(x,feat_q)
        x = self.block2(x)
        feat_q = self.cluster2(x, feat_q)
        x = self.block3(x)
        feat_q = self.cluster3(x, feat_q)
        # x = self.block4(x)
        # feat_q = self.cluster4(x, feat_q)

        res_clusters = self.cluster_out(self.out_proj(x),feat_q)
        #res_clusters = self.instance_norm(res_clusters)
        noise_feat = self.noise_proj(res_clusters[:,self.query_size//4*3:,::])
        clear_feat = res_clusters[:,:self.query_size//4*3,::]
        noise_res = self.clear_head(noise_feat+clear_feat)
        clear_res = self.clear_head(clear_feat)
        return noise_res,clear_res,noise_feat,clear_feat


