from turtle import down
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

IM_MEAN = [tensor(0.1009), tensor(0.1462), tensor(0.0698), tensor(0.1462)]
IM_STD = [tensor(1.0786), tensor(1.3015), tensor(0.9083), tensor(1.3028)]
GT_MEAN = [tensor(0.0559), tensor(0.0937), tensor(0.0327), tensor(0.0937)]
GT_STD =  [tensor(0.0697), tensor(0.1044), tensor(0.0477), tensor(0.1044)]


def pack_raw(raw):
    # pack Bayer image to 3 channels
    #im = raw.raw_image_visible.astype(np.float32)
    #CHANGES NOT TESTED
    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    return im

    #im = np.expand_dims(im, axis=2)
    #img_shape = im.shape
    #H = img_shape[0]
    #W = img_shape[1]

    #out = np.concatenate((im[0:H:2, 0:W:2, :],
    #                      im[0:H:2, 1:W:2, :],
    #                      im[1:H:2, 1:W:2, :],
    #                      im[1:H:2, 0:W:2, :]), axis=2)


    #return out



NUM_CHANNELS = 4
'''
Get the statistics of GT images and of low exposure images and use these for dataset normalization
'''
def get_statistics(dir_path = "../Sony"):
    
    gt_files = os.listdir(os.path.join(dir_path, "long/"))
    short_files = os.listdir(os.path.join(dir_path, "short/"))

    ids = [id.split("_")[0] for id in gt_files]
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

def create_dataset(dir_path = "../Sony", train_ratio = 0.5):
    gt_files = os.listdir(os.path.join(dir_path, "long/"))
    short_files = os.listdir(os.path.join(dir_path, "short/"))

    ids = [id.split("_")[0] for id in gt_files]
    n = len(ids)

    train_ids = random.sample(ids, int(n * train_ratio))
    test_ids = [id for id in ids if id not in train_ids]
    #POSSIBLE: add validation set as well

    return VideDataset(dir_path, train_ids), VideDataset(dir_path, test_ids)



class VideDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path = "../Sony", ids = [], crop_size=512, downsampling_ratio=4):
        super().__init__()
        self.ids = ids

        self.crop_size = crop_size
        self.a = downsampling_ratio

        self.dir_path = dir_path
        self.gt_files = os.listdir(os.path.join(dir_path, "long/"))
        self.short_files = os.listdir(os.path.join(dir_path, "short/"))

        #remove the following line
        #self.ids = [id.split("_")[0] for id in self.gt_files]
        
        self.ids.sort()

    def __getitem__(self, index):
        id = self.ids[index]
        
        #finding gt file and loading it
        names = [name for name in self.gt_files if name.startswith(id)]
        gtpath = os.path.join(self.dir_path, f"long/{names[0]}")
        f = open(gtpath,  'rb')
        raw = rawpy.imread(f)

        gtimage = torch.from_numpy(np.expand_dims(pack_raw(raw), axis=0)).permute(-1, 1, 2, 0)
        gtimage = gtimage.reshape(gtimage.shape[:-1])
        gt_exposure = float(names[0][9:-5])
        
        #load

        #finding low exposure files and loading two random frames
        names = [name for name in self.short_files if name.startswith(id)]
        frames = random.sample(names, 2)
        paths = (os.path.join(self.dir_path, f"short/{frames[0]}"),
        os.path.join(self.dir_path, f"short/{frames[1]}"))
        images = []
        for i,impath in enumerate(paths):
            f = open(impath,  'rb')
            raw = rawpy.imread(f)
            
            im_exposure = float(frames[i][9:-5])
            ratio = min(gt_exposure / im_exposure, 300)

            images.append(torch.from_numpy(np.expand_dims(pack_raw(raw), axis=0)).permute(-1, 1, 2, 0).reshape(gtimage.shape)*ratio)
            
            images[i] = images[i].permute(1,2,0)[::self.a, ::self.a].permute(2,0,1) #this line performs down-sampling by a ratio (NOT TESTED)

        #Normalization preprocessing
        t1 = torchvision.transforms.Normalize(IM_MEAN, IM_STD) 
        t2 = torchvision.transforms.Normalize(GT_MEAN, GT_STD) 
        crop = torchvision.transforms.RandomCrop(self.crop_size) #size is changeable we can downsample the image to reduce image size
        t1 = torchvision.transforms.Compose([t1, crop])
        t2 = torchvision.transforms.Compose([t2, crop])

        return t1(images[0]), t1(images[1]), t2(gtimage)

    def __len__(self):
        return len(self.ids)

if __name__ == "__main__":
    #tr, ts = create_dataset()
    #print(len(ts))
    print(get_statistics())