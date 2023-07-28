from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
import torch
import torch.nn.functional as F
import random
import cv2
class TrainDataSet(Dataset):
    def __init__(self,syn_rain_path,syn_ground_path,img_size):
        super(TrainDataSet,self).__init__()
        self.img_size = img_size
        self.syn_rain_path = syn_rain_path
        self.syn_background_path = syn_ground_path
        self.img_syn = os.listdir(self.syn_rain_path)
        self.img_background = os.listdir(self.syn_background_path)
        self.paired_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.syn_size = len(self.img_syn)
        self.size = self.syn_size
    def __getitem__(self, index):
        imgSyn = Image.open(os.path.join(self.syn_rain_path,self.img_syn[index])).convert('RGB')
        imgB = Image.open(os.path.join(self.syn_background_path,self.img_syn[index])).convert('RGB')
        if np.random.random() < 0.5:
            imgSyn= imgSyn.transpose(Image.FLIP_LEFT_RIGHT)
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
        w,h = imgSyn.size
        rand_flag = np.random.random()
        if rand_flag < 0.5:
            if w>self.img_size and h>self.img_size:
                dw = w-self.img_size
                dh = h-self.img_size
                ws = np.random.randint(dw+1)
                hs = np.random.randint(dh+1)
                imgSyn = imgSyn.crop((ws, hs, ws+self.img_size, hs+self.img_size))
                imgB = imgB.crop((ws, hs, ws + self.img_size, hs + self.img_size))
        imgSyn = self.paired_transform(imgSyn)
        imgB = self.paired_transform(imgB)
        return [imgB,imgSyn]
    def __len__(self):
        return self.size



class DenoiseDataSet(Dataset):
    def __init__(self,syn_rain_path,syn_ground_path,img_size):
        super(DenoiseDataSet,self).__init__()
        self.img_size = img_size
        self.syn_background_path = syn_ground_path
        self.img_syn = os.listdir(self.syn_background_path)
        self.img_background = os.listdir(self.syn_background_path)
        self.paired_transform_1 = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.paired_transform_2 = transforms.Compose([
            transforms.RandomCrop((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # self.paired_transform_1 = transforms.Compose([
        #     transforms.Resize((self.img_size, self.img_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5])
        # ])
        # self.paired_transform_2 = transforms.Compose([
        #     transforms.RandomCrop((self.img_size, self.img_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5])
        # ])
        self.noise_levels = [15,25,50]
        self.syn_size = len(self.img_syn)
        self.size = self.syn_size
    def __getitem__(self, index):
        imgB = Image.open(os.path.join(self.syn_background_path,self.img_syn[index])).convert('RGB')
        #imgB = Image.open(os.path.join(self.syn_background_path, self.img_syn[index])).convert('L')
        if np.random.random() < 0.5:
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
        rand_flag = np.random.random()
        if rand_flag < 0.5:
            imgB = self.paired_transform_1(imgB)
        else:
            imgB = self.paired_transform_2(imgB)
        imgSyn = imgB.clone()
        noise = torch.randn(imgSyn.size()).mul_(random.choice(self.noise_levels) / 255.0)
        imgSyn = imgSyn + noise
        return [imgB,imgSyn]
    def __len__(self):
        return self.size


class ISTDTrainDataSet(Dataset):
    def __init__(self,syn_rain_path,syn_ground_path,img_size):
        super(ISTDTrainDataSet,self).__init__()
        self.img_size = img_size
        self.syn_rain_path = syn_rain_path
        self.syn_background_path = syn_ground_path
        self.img_syn = os.listdir(self.syn_rain_path)
        self.img_background = os.listdir(self.syn_background_path)
        self.paired_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.syn_size = len(self.img_syn)
        self.size = self.syn_size
    def __getitem__(self, index):
        imgSyn = Image.open(os.path.join(self.syn_rain_path,self.img_syn[index])).convert('RGB')
        imgB = Image.open(os.path.join(self.syn_background_path,self.img_syn[index])).convert('RGB')
        if np.random.random() < 0.5:
            imgSyn= imgSyn.transpose(Image.FLIP_LEFT_RIGHT)
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
        imgSyn = self.paired_transform(imgSyn)
        imgB = self.paired_transform(imgB)

        return [imgB,imgSyn]
    def __len__(self):
        return self.size


class TestDataSet(Dataset):
    def __init__(self,syn_rain_path,syn_ground_path,img_size=256):
        super(TestDataSet,self).__init__()
        self.img_size = img_size
        self.syn_rain_path = syn_rain_path
        self.syn_background_path = syn_ground_path
        self.img_syn = os.listdir(self.syn_rain_path)
        self.img_background = os.listdir(self.syn_background_path)
        self.paired_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.syn_size = len(self.img_syn)
        self.size = self.syn_size
    def __getitem__(self, index):
        imgSyn = Image.open(os.path.join(self.syn_rain_path,self.img_syn[index])).convert('RGB')
        imgB = Image.open(os.path.join(self.syn_background_path,self.img_syn[index])).convert('RGB')
        imgSyn = imgSyn.resize(imgB.size)
        imgB = self.paired_transform(imgB)
        imgSyn = self.paired_transform(imgSyn)
        return [imgB,imgSyn]
    def __len__(self):
        return self.size



class ITSTrainDataSet(Dataset):
    def __init__(self,syn_rain_path,syn_ground_path,img_size):
        super(ITSTrainDataSet,self).__init__()
        self.img_size = img_size
        self.data_path = syn_rain_path
        self.gt_path = syn_ground_path
        self.data_names = os.listdir(self.data_path)
        self.paired_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.size = 1000
    def __getitem__(self, index):
        data_name = self.data_names[index]
        data_id = data_name.split('_')[0]
        gt_name = data_id+'.jpg'
        data = Image.open(os.path.join(self.data_path,data_name)).convert('RGB')
        gt = Image.open(os.path.join(self.gt_path,gt_name)).convert('RGB')
        data = data.resize(gt.size)
        if np.random.random() < 0.5:
            data= data.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        w,h = gt.size
        rand_flag = np.random.random()
        if rand_flag < 0.5:
            if w>self.img_size and h>self.img_size:
                dw = w-self.img_size
                dh = h-self.img_size
                ws = np.random.randint(dw+1)
                hs = np.random.randint(dh+1)
                data = data.crop((ws, hs, ws+self.img_size, hs+self.img_size))
                gt = gt.crop((ws, hs, ws + self.img_size, hs + self.img_size))
        data = self.paired_transform(data)
        gt = self.paired_transform(gt)
        return [gt, data]
    def __len__(self):
        return self.size

class NH_HAZE_Dataset(Dataset):
    def __init__(self, syn_rain_path,img_size=256):
        super(NH_HAZE_Dataset, self).__init__()
        self.img_size = img_size
        self.syn_rain_path = syn_rain_path
        self.img_syn = []
        for name in os.listdir(self.syn_rain_path):
            id = name.split('_')[0]
            self.img_syn.append(id)
        self.img_syn = list(set(self.img_syn))
        self.paired_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.syn_size = len(self.img_syn)
        self.size = self.syn_size

    def __getitem__(self, index):
        id = self.img_syn[index]
        data_name = id+'_hazy.png'
        gt_name = id+'_GT.png'
        data = Image.open(os.path.join(self.syn_rain_path, data_name)).convert('RGB')
        gt = Image.open(os.path.join(self.syn_rain_path, gt_name)).convert('RGB')
        data = data.resize(gt.size)
        if np.random.random() < 0.5:
            data= data.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        w,h = gt.size
        rand_flag = np.random.random()
        if rand_flag < 0.5:
            if w>self.img_size and h>self.img_size:
                dw = w-self.img_size
                dh = h-self.img_size
                ws = np.random.randint(dw+1)
                hs = np.random.randint(dh+1)
                data = data.crop((ws, hs, ws+self.img_size, hs+self.img_size))
                gt = gt.crop((ws, hs, ws + self.img_size, hs + self.img_size))
        imgB = self.paired_transform(gt)
        imgSyn = self.paired_transform(data)
        return [imgB, imgSyn]

    def __len__(self):
        return self.size

class DenseHazyDataset(Dataset):
    def __init__(self, syn_rain_path,gt_path,img_size=256):
        super(DenseHazyDataset, self).__init__()
        self.img_size = img_size
        self.syn_rain_path = syn_rain_path
        self.gt_path = gt_path
        self.img_syn = []
        for name in os.listdir(self.syn_rain_path):
            id = name.split('_')[0]
            self.img_syn.append(id)
        self.img_syn = list(set(self.img_syn))
        self.paired_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.syn_size = len(self.img_syn)
        self.size = self.syn_size

    def __getitem__(self, index):
        id = self.img_syn[index]
        data_name = id+'_hazy.png'
        gt_name = id+'_GT.png'
        imgSyn = Image.open(os.path.join(self.syn_rain_path, data_name)).convert('RGB')
        imgB = Image.open(os.path.join(self.gt_path, gt_name)).convert('RGB')
        imgSyn = imgSyn.resize(imgB.size)
        imgB = self.paired_transform(imgB)
        imgSyn = self.paired_transform(imgSyn)
        return [imgB, imgSyn]

    def __len__(self):
        return self.size

class SOTODataSet(Dataset):
    def __init__(self,syn_rain_path,syn_ground_path,img_size=256):
        super(SOTODataSet,self).__init__()
        self.img_size = img_size
        self.data_path = syn_rain_path
        self.gt_path = syn_ground_path
        self.data_names = os.listdir(self.data_path)
        self.paired_transform = transforms.Compose([
            transforms.CenterCrop((self.img_size, self.img_size)),
            #transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.size = len(self.data_names)
    def __getitem__(self, index):
        data_name = self.data_names[index]
        data_id = data_name.split('_')[0]
        gt_name = data_id+'.png'
        data = Image.open(os.path.join(self.data_path,data_name)).convert('RGB')
        gt = Image.open(os.path.join(self.gt_path,gt_name)).convert('RGB')
        #data = data.resize(gt.size)
        data = self.paired_transform(data)
        gt = self.paired_transform(gt)
        return [gt, data]
    def __len__(self):
        return self.size


class GoProTrainDataSet(Dataset):
    def __init__(self,syn_rain_path,syn_ground_path,img_size):
        super(GoProTrainDataSet,self).__init__()
        self.img_size = img_size
        self.data_path = syn_rain_path
        self.folders = os.listdir(self.data_path)
        #self.data = [os.listdir(os.path.join(data_path,folder,'blur')) for folder in self.folders]
        self.img_path = []
        for folder in self.folders:
            folder_path = os.path.join(self.data_path,folder,'blur')
            img_names = os.listdir(folder_path)
            for img_name in img_names:
                img_path = os.path.join(folder_path,img_name)
                #b_path = img_path.replace('blur','sharp')
                self.img_path.append(img_path)

        self.paired_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.size = len(self.img_path)
        print('size of dataset = %d'%self.size)
    def __getitem__(self, index):
        imgSyn = Image.open(self.img_path[index]).convert('RGB')
        imgB = Image.open(self.img_path[index].replace('blur','sharp')).convert('RGB')
        if np.random.random() < 0.5:
            imgSyn= imgSyn.transpose(Image.FLIP_LEFT_RIGHT)
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
        w,h = imgSyn.size
        rand_flag = np.random.random()
        if rand_flag < 0.5:
            if w>self.img_size and h>self.img_size:
                dw = w-self.img_size
                dh = h-self.img_size
                ws = np.random.randint(dw+1)
                hs = np.random.randint(dh+1)
                imgSyn = imgSyn.crop((ws, hs, ws+self.img_size, hs+self.img_size))
                imgB = imgB.crop((ws, hs, ws + self.img_size, hs + self.img_size))
        imgSyn = self.paired_transform(imgSyn)
        imgB = self.paired_transform(imgB)

        return [imgB,imgSyn]
    def __len__(self):
        return self.size


class AllInOneDataSet(Dataset): #100个epoch，30降一次
    def __init__(self,syn_rain_path,syn_ground_path,img_size):
        self.noise_imgs_root = ''

        self.rain_img_root = ''
        self.rain_gt_root = ''

        self.haze_img_root = ''
        self.haze_gt_root = ''

        self.img_size = img_size
        self.noise_imgs = os.listdir(self.noise_imgs_root)
        self.rain_img = os.listdir(self.rain_img_root)
        self.rain_gt = os.listdir(self.rain_gt_root)
        self.haze_img = os.listdir(self.haze_img_root)
        self.haze_gt = os.listdir(self.haze_gt_root)

        self.noise_levels = [15,25,50]

        self.len_noise = len(self.noise_imgs)
        self.len_rain = len(self.rain_img)
        self.len_haze = len(self.haze_img)

        self.size = 7000
        self.paired_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
        ])
        self.paired_transform_2 = transforms.Compose([
            transforms.RandomCrop((self.img_size,self.img_size)),
            transforms.ToTensor(),
        ])
        print(self.size)
    def __getitem__(self, index):
        i = index%3
        if i==0:
            return self.get_noise_data(index)
        elif i == 1:
            return self.get_rain_data(index)
        else:
            return self.get_haze_data(index)
    def get_noise_data(self,index):
        imgB = Image.open(os.path.join(self.noise_imgs_root,self.noise_imgs[index//3%self.len_noise])).convert('RGB')
        #imgB = Image.open(os.path.join(self.syn_background_path, self.img_syn[index])).convert('L')
        if np.random.random() < 0.5:
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
        rand_flag = np.random.random()
        if rand_flag < 0.5:
            imgB = self.paired_transform(imgB)
        else:
            imgB = self.paired_transform_2(imgB)
        imgSyn = imgB.clone()
        noise = torch.randn(imgSyn.size()).mul_(random.choice(self.noise_levels) / 255.0)
        imgSyn = imgSyn + noise
        return [imgB,imgSyn]
    def get_rain_data(self, index):
        imgSyn = Image.open(os.path.join(self.rain_img_root,self.rain_img[index//3%self.len_rain])).convert('RGB')
        imgB = Image.open(os.path.join(self.rain_gt_root,self.rain_gt[index//3%self.len_rain])).convert('RGB')
        if np.random.random() < 0.5:
            imgSyn= imgSyn.transpose(Image.FLIP_LEFT_RIGHT)
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
        w,h = imgSyn.size
        rand_flag = np.random.random()
        if rand_flag < 0.5:
            if w>self.img_size and h>self.img_size:
                dw = w-self.img_size
                dh = h-self.img_size
                ws = np.random.randint(dw+1)
                hs = np.random.randint(dh+1)
                imgSyn = imgSyn.crop((ws, hs, ws+self.img_size, hs+self.img_size))
                imgB = imgB.crop((ws, hs, ws + self.img_size, hs + self.img_size))
        imgSyn = self.paired_transform(imgSyn)
        imgB = self.paired_transform(imgB)
        return [imgB,imgSyn]
    def get_haze_data(self, index):
        data_name = self.haze_img[index//3%self.len_haze]
        data_id = data_name.split('_')[0]
        gt_name = data_id + '.jpg'
        data = Image.open(os.path.join(self.haze_img_root, data_name)).convert('RGB')
        gt = Image.open(os.path.join(self.haze_gt_root, gt_name)).convert('RGB')
        data = data.resize(gt.size)
        if np.random.random() < 0.5:
            data = data.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = gt.size
        rand_flag = np.random.random()
        if rand_flag < 0.5:
            if w > self.img_size and h > self.img_size:
                dw = w - self.img_size
                dh = h - self.img_size
                ws = np.random.randint(dw + 1)
                hs = np.random.randint(dh + 1)
                data = data.crop((ws, hs, ws + self.img_size, hs + self.img_size))
                gt = gt.crop((ws, hs, ws + self.img_size, hs + self.img_size))
        data = self.paired_transform(data)
        gt = self.paired_transform(gt)
        return [gt, data]

    def __len__(self):
        return self.size


# class CelebaTrainDataSet(Dataset):
#     def __init__(self,syn_rain_path,syn_ground_path,img_size=256):
#         super(CelebaTrainDataSet,self).__init__()
#         self.img_size = img_size
#         self.syn_rain_path = syn_rain_path
#         self.img_syn = os.listdir(self.syn_rain_path)[3000:]
#         self.paired_transform = transforms.Compose([
#             transforms.Resize((self.img_size,self.img_size)),
#             transforms.ToTensor(),
#         ])
#         self.tramsform_n = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         self.syn_size = len(self.img_syn)
#         self.size = self.syn_size
#     def __getitem__(self, index):
#         imgB = Image.open(os.path.join(self.syn_rain_path,self.img_syn[index])).convert('RGB')
#         mask = generate_stroke_mask(im_size=(self.img_size,self.img_size),parts=np.random.randint(2,8))[0]
#         if np.random.random() < 0.5:
#             imgB= imgB.transpose(Image.FLIP_LEFT_RIGHT)
#         imgB = self.paired_transform(imgB)#[0,1]
#         imgdata = (imgB+mask).clamp(0,1)
#         imgdata = self.tramsform_n(imgdata)
#         imgB = self.tramsform_n(imgB)
#         return [imgB,imgdata]
#     def __len__(self):
#         return self.size




