from audioop import add
import torch
import torch.nn as nn
import torchvision
import scipy.io


MEAN_VALUES = torch.tensor([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3)) #mean values of the dataset

class Vgg19(nn.Module):
    def __init__(self):
        #load vgg model and turn off layers from conv 4_3 and on (stop is at index 23, last layer is index 22)
        #The papaer uses a pretrained model
        features = list(torchvision.models.vgg19(pretrained=True).features)
        
        #The paper uses avg pooling which is incosistent with the VGG architecture
        block1 = nn.Sequential(*features[:4])
        block2 = nn.Sequential(*features[4:9])
        block3 = nn.Sequential(*features[9:18])
        block4 = nn.Sequential(*features[18:23])

        #TODO: find dataset mean and std (could normalize in dataset loader)
        #self.norm = torchvision.transforms.Normalize() 

        self.blocks = [block1, block2, block3, block4]

    def forward(self, x):
        res =  []
        x = x - MEAN_VALUES #should use a normalization layer instead of substitution as in paper
        for b in self.blocks:
            res.append(x)
            x =  b(x)
        res.append(x)
        return res


def compute_error(real,fake):
    return torch.mean(torch.abs(fake-real))


def F_loss(real,fake):
    # p0=compute_error(real[0],fake[0])
    # p1=compute_error(real[1],fake[1])
    # p2=compute_error(real[2],fake[2])
    # p3=compute_error(real[3],fake[3])
    # p4=compute_error(real[4],fake[4])
    return torch.sum([compute_error(real[i], fake[i]) for i in range(5)])