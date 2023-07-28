## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Attention1x1(nn.Module):
    def __init__(self, query_size,bias=False):
        super(Attention1x1, self).__init__()
        self.num_heads = 1
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qkv = nn.Conv2d(query_size, query_size * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(query_size * 3, query_size * 3, kernel_size=1, stride=1,groups=query_size * 3, bias=bias)
        self.project_out = nn.Conv2d(query_size, query_size, kernel_size=1, bias=bias)

    def forward(self, x):
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        b, c, h, w = v.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=h)
        out = self.project_out(out)
        return out

class CTCA(nn.Module):
    def __init__(self, dim,query_size,bias=False,hw=128,pool_V=True):
        super(CTCA, self).__init__()
        num_heads = 1
        self.pool_V = pool_V
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.hw = hw
        self.pool = nn.AdaptiveAvgPool2d((hw,hw))
        self.project_out = nn.Conv2d(query_size, query_size, kernel_size=1, bias=bias)


    def forward(self, x,q):
        #kv = self.kv_dwconv(self.kv(x))
        #k, v = kv.chunk(2, dim=1)
        k = x
        v = x
        k = F.adaptive_avg_pool2d(k, (self.hw, self.hw))
        if self.pool_V:
            v = F.adaptive_avg_pool2d(v,(self.hw,self.hw))
        b,c,h,w = v.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=2)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


##########################################################################
class ChannelTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(ChannelTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class SubAttention(nn.Module):
    def __init__(self,in_channels=128,out_channels=32,ratio=1,bias=False):
        super(SubAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.qk = nn.Conv2d(in_channels=in_channels,out_channels=out_channels*2,
                            kernel_size=ratio,stride=ratio,bias=bias)
        self.v = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                           kernel_size=1,stride=1)
    def forward(self,x):# BCHW
        qk = self.qk(x)
        v = self.v(x)
        q = qk[:,self.out_channels:,::]
        k = qk[:,:self.out_channels,::]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=1)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=1)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        return out


class MultiScaleAttention(nn.Module):
    def __init__(self, dim=128,bias=False,scales=[1,2,4,8]):
        super(MultiScaleAttention, self).__init__()
        self.num_heads = 4
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.subatt0 = SubAttention(in_channels=dim,out_channels=dim//4,ratio=scales[0],bias=bias)
        self.subatt1 = SubAttention(in_channels=dim, out_channels=dim // 4, ratio=scales[1], bias=bias)
        self.subatt2 = SubAttention(in_channels=dim, out_channels=dim // 4, ratio=scales[2], bias=bias)
        self.subatt3 = SubAttention(in_channels=dim, out_channels=dim // 4, ratio=scales[3], bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape
        x0 = self.subatt0(x)
        x1 = self.subatt1(x)
        x2 = self.subatt2(x)
        x3 = self.subatt3(x)
        out = torch.cat([x0,x1,x2,x3],dim=1)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)+x
        return out


##########################################################################
class MultiScaleChannelTransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type,scales):
        super(MultiScaleChannelTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MultiScaleAttention(dim, bias,scales)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x




