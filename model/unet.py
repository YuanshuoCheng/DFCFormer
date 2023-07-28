import torch
from model.parts import MobileConv,DownBlock,UpBlock,OutConv
import torch.nn as nn
#from parts import MobileConv,DownBlock,UpBlock,OutConv


class UNET(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UNET, self).__init__()
        self.head = MobileConv(in_channels=in_channels,out_channels=32)
        self.tail = MobileConv(32,16)
        self.out = OutConv(16,out_channels)
        self.encoderBlock0 = DownBlock(32,64)
        self.encoderBlock1 = DownBlock(64,128)
        self.encoderBlock2 = DownBlock(128,256)
        self.encoderBlock3 = DownBlock(256,256)
        self.decoderBlock0 = UpBlock(512,128)
        self.decoderBlock1 = UpBlock(256,64)
        self.decoderBlock2 = UpBlock(128,32)
        self.decoderBlock3 = UpBlock(64,32)
    def forward(self,x):
        x = self.head(x)
        x,feat0 = self.encoderBlock0(x)
        x, feat1 = self.encoderBlock1(x)
        x, feat2 = self.encoderBlock2(x)
        x, feat3 = self.encoderBlock3(x)

        x = self.decoderBlock0(x,feat3)
        x = self.decoderBlock1(x, feat2)
        x = self.decoderBlock2(x, feat1)
        x = self.decoderBlock3(x, feat0)
        x = self.tail(x)
        return self.out(x)


if __name__ == '__main__':
    with torch.no_grad():
        image = torch.randn((1,3,301,256)).cuda()
        net = UNET(3,3).cuda()
        res = net(image)
        print(res.shape)

