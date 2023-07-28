from model.model import DFCFormer
import torch
from data.datasets import TestDataSet
from torch.utils.data.dataloader import DataLoader
import os
from utils import tensor2im
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

save_path = './res_imgs'
if not os.path.exists(save_path):
    os.makedirs(save_path)

test_set = TestDataSet(syn_path= '',
                        ground_path='')
test_loader = DataLoader(test_set, batch_size=1,
                          num_workers=1, drop_last=False, shuffle=False, pin_memory=False)
model = DFCFormer().eval().cuda()
model.load_state_dict(torch.load('./weights/final.pth'))

avg_ssim = 0
avg_panr = 0
cnt = 0
with torch.no_grad():
    for imgB,imgSyn,img_name in test_loader:
        imgB = imgB.cuda()
        imgSyn = imgSyn.cuda()
        res = model(imgSyn)
        res = tensor2im(res[0].detach().cpu())
        gt = tensor2im(imgB[0].detach().cpu())

        res = res[:,:,0]
        gt = gt[:,:,0]
        psnr = peak_signal_noise_ratio(gt, res)
        avg_panr += psnr
        ssim = structural_similarity(gt, res, multichannel=False)
        avg_ssim += ssim
        cnt+=1
        cv2.imwrite(os.path.join(save_path, img_name[0]), res)
        if cnt%10 == 0:
            print(cnt)

print('PSNR: %.4f'%(avg_panr/cnt))
print('SSIM %.4f'%(avg_ssim/cnt))




