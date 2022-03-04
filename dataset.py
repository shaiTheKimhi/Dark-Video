
import cv2
import torch
import os
import torchvision
from torchvision import transforms
import numpy as np
import random
import rawpy

import tqdm

from torch import tensor

IM_MEAN = [tensor(19.7701), tensor(35.1723), tensor(26.9472)]
IM_STD =  [tensor(11.1887), tensor(28.1479), tensor(17.9918)]
GT_MEAN = [tensor(60.1718), tensor(72.0474), tensor(81.7679)]
GT_STD =  [tensor(38.5039), tensor(43.2455), tensor(48.7318)]



def pack_raw(raw, gt=True):
    # pack Bayer image to 3 channels
    #im = raw.raw_image_visible.astype(np.float32)
    #CHANGES NOT TESTED
    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    #using debayer either at start or at the end

    #im = raw.raw_image_visible.astype(np.float32)
    if gt:
        #im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    else:
        
        im = im.astype(np.int16)
    #im = np.expand_dims(im, axis=2) removed cause of next line
    return torch.tensor(im)#.expand([3] + list(im.shape)).permute(1,2,0)




NUM_CHANNELS = 3 #RGB 1 for RAW DATASET
'''
Get the statistics of GT images and of low exposure images and use these for dataset normalization
'''
def get_statistics(dir_path = "../", RGB=True):
    if not RGB:
        pass #here we will use RAW dataset
    ids = []
    with open(os.path.join(dir_path, "test_list.txt"), "r") as file:
        ids += file.readlines()
    
    with open(os.path.join(dir_path, "train_list.txt"), "r") as file:
        ids += file.readlines()

    ids = [i.replace('\n','') for i in ids]
    ids.sort()
    n = len(ids)

    marr = torch.zeros(n, NUM_CHANNELS) #3 is channels number
    stdarr = torch.zeros(n, NUM_CHANNELS)

    gtmarr = torch.zeros(n, NUM_CHANNELS)
    gtstdarr = torch.zeros(n, NUM_CHANNELS)

    dataset = VideDataset(dir_path, ids)
    for i in tqdm.tqdm(range(len(dataset))):
        #TODO: add statistics calculations here
        x1, x2, y = dataset[i]
        x1, x2, y = x1.float(), x2.float(), y.float()

        m = torch.tensor([torch.mean(x1[j]) for j in range(NUM_CHANNELS)])/2
        std = torch.tensor([torch.std(x1[j]) for j in range(NUM_CHANNELS)])/2

        m += torch.tensor([torch.mean(x2[j]) for j in range(NUM_CHANNELS)])/2
        std += torch.tensor([torch.std(x2[j]) for j in range(NUM_CHANNELS)])/2

        for j in range(NUM_CHANNELS):
            marr[i][j] += m[j]
            stdarr[i][j] += std[j]

        m = torch.tensor([torch.mean(y[j]) for j in range(NUM_CHANNELS)])
        std = torch.tensor([torch.std(y[j]) for j in range(NUM_CHANNELS)])

        for j in range(NUM_CHANNELS):
            gtmarr[i][j] += m[j]
            gtstdarr[i][j] += std[j]

    return [torch.mean(marr.T[i]) for i in range(NUM_CHANNELS)], [torch.mean(stdarr.T[i]) for i in range(NUM_CHANNELS)],\
    [torch.mean(gtmarr.T[i]) for i in range(NUM_CHANNELS)], [torch.mean(gtstdarr.T[i]) for i in range(NUM_CHANNELS)]

def create_dataset(dir_path = "../", train_ratio = 0.5, RGB=True):
    if not RGB:
        pass #here we will use the RAW dataset
    with open(os.path.join(dir_path, "test_list.txt"), "r") as file:
        test_ids = file.readlines()
    
    with open(os.path.join(dir_path, "train_list.txt"), "r") as file:
        train_ids = file.readlines()

    test_ids = [i.replace('\n','') for i in test_ids]
    train_ids = [i.replace('\n','') for i in train_ids]

    #POSSIBLE: add validation set as well

    return VideDataset(dir_path, train_ids), VideDataset(dir_path, test_ids)



class VideDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path = "../", ids = [], crop_size=512, downsampling_ratio=1):
        super().__init__()
        self.ids = ids

        self.crop_size = crop_size
        self.a = downsampling_ratio

        self.gt_dir_path = os.path.join(dir_path, "long") #long exposure path   
        self.dir_path = os.path.join(dir_path, "VBM4D_rawRGB") #short exposure path
        self.short_files = os.listdir(self.dir_path)
        
        self.ids.sort()

    def __getitem__(self, index):
        id = self.ids[index]
        
        #finding gt file and loading it
        #names = [name for name in self.gt_files if name.startswith(id)]
        gtpath = os.path.join(self.gt_dir_path, id)
        names = [name for name in os.listdir(gtpath) if name.endswith('png') and name.startswith('half')] #get png file, choose half resolution or full resolution
        
        gtimage = torch.from_numpy(cv2.imread(os.path.join(gtpath, names[0]))).permute(-1,0,1) #channels at the start

        gtshape = gtimage.shape
        gtimage = gtimage.permute(1,2,0)[::self.a, ::self.a].permute(2,0,1) #this line performs down-sampling by a ratio
        
        gtimage = gtimage.float()
        
        #load

        #finding low exposure files and loading two random frames
        dir_path = os.path.join(self.dir_path, id)
        names = [name for name in os.listdir(dir_path) if name.endswith('png')]
        frames = random.sample(names, 2)
        paths = (os.path.join(dir_path, f"{frames[0]}"),
        os.path.join(dir_path, f"{frames[1]}"))
        images = []
        for i,impath in enumerate(paths):
            #print(impath)
            #exit(0)

            im = torch.from_numpy(cv2.imread(impath)).permute(-1,0,1)

            im = im.permute(1,2,0)[::self.a, ::self.a].permute(2,0,1).float() #this line performs down-sampling by a ratio
            images.append(im)

        
        #return images[0], images[1], gtimage
        
        #crop = torchvision.transforms.RandomCrop(self.crop_size) #size is changeable we can downsample the image to reduce image size,
        #crop without random crop as it is not consistent
        sx = torch.randint(0, gtimage.shape[1]-self.crop_size, (1,)).item()
        sy = torch.randint(0, gtimage.shape[2]-self.crop_size, (1,)).item()
        gtimage = gtimage[::, sx:sx + self.crop_size, sy:sy + self.crop_size]
        images = [image[::, sx:sx+self.crop_size, sy:sy+self.crop_size] for image in images]
        
        #RAW dataset
        #t1 = torchvision.transforms.Compose([torchvision.transforms.Normalize([tensor(115143.6250)], [tensor(14786.9463)]), crop])
        #t2 = torchvision.transforms.Compose([torchvision.transforms.Normalize([tensor(0.0619)], [tensor(0.0695)]), crop])
        #return t1(images[0]), t1(images[1]), t2(gtimage)

        #Normalization preprocessing
        t1 = torchvision.transforms.Normalize(IM_MEAN, IM_STD) 
        t2 = torchvision.transforms.Normalize(GT_MEAN, GT_STD) 

        return t1(images[0]), t1(images[1]), t2(gtimage) 

    def __len__(self):
        return len(self.ids)

if __name__ == "__main__":
    #tr, ts = create_dataset()
    #print(len(ts))
    print(get_statistics())
