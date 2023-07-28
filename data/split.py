from PIL import Image


def cal_offset(s,patch_size):
    off_set = []
    cat_n = patch_size//s+1 if patch_size%s!=0 else patch_size//s
    s = cat_n*s

    split_n = s // patch_size + 1 if s%patch_size!=0 else s//patch_size
    for i in range(split_n - 1):
        off_set.append([i * patch_size, i * patch_size + patch_size])
    off_set.append([s - patch_size, s])
    return off_set,cat_n

def cat(img,cat_n_w,cat_n_h):
    w,h = img.size
    target_w = cat_n_w*w
    target_h = cat_n_h*h
    background = Image.new('RGB', (target_w,target_h))
    for i in range(cat_n_w):
        for j in range(cat_n_h):
            background.paste(img,(i*w,j*h))
    return background

def split(img,off_set_w,off_set_h):
    images = []
    for i in off_set_w:
        for j in off_set_h:
            l,r = i
            t,b = j
            images.append(img.crop([l,t,r,b]))
    return images



def cat_split(img,patch_size=256):
    w,h = img.size
    off_set_w,cat_n_w = cal_offset(w,patch_size)
    off_set_h,cat_n_h = cal_offset(h,patch_size)
    background = cat(img,cat_n_w,cat_n_h)
    images = split(background,off_set_w,off_set_h)
    return images,off_set_w,off_set_h

def recover(images,off_set_w,off_set_h,ori_size):
    target_w = off_set_w[-1][-1]
    target_h = off_set_h[-1][-1]
    background = Image.new('RGB', (target_w,target_h))
    cnt = 0
    for i in off_set_w:
        for j in off_set_h:
            background.paste(images[cnt],(i[0],j[0]))
            cnt+=1
    return background.crop([0,0,ori_size[0],ori_size[1]])

if __name__ == '__main__':
    image = Image.open('../000001.png')
    images,off_set_w,off_set_h = cat_split(image,256)
    print('图像大小：',image.size)
    print('分割数量：',len(images))
    print('patch大小：',images[0].size)
    recovered = recover(images,off_set_w,off_set_h,ori_size=image.size)
    recovered.show()



