from audioop import add
#from turtle import forward
import torch
import torch.nn as nn
import torchvision
import scipy.io



class Vgg19(nn.Module):
    def __init__(self):
        #load vgg model and turn off layers from conv 4_3 and on (stop is at index 23, last layer is index 22)
        #The papaer uses a pretrained model
        super(Vgg19, self).__init__()
        features = list(torchvision.models.vgg19(pretrained=True).features)
        
        #The paper uses avg pooling which is incosistent with the VGG architecture
        self.block1 = nn.Sequential(*features[:4])
        self.block2 = nn.Sequential(*features[4:9])
        self.block3 = nn.Sequential(*features[9:18])
        self.block4 = nn.Sequential(*features[18:23])

        #self.blocks = [block1, block2, block3, block4]

    def forward(self, x):
        #x = x.float()
        blocks = [self.block1, self.block2, self.block3, self.block4]
        res =  []
        for b in blocks:
            res.append(x)
            x =  b(x)
        res.append(x)
        return res


def compute_error(real,fake):
    return torch.mean(torch.abs(fake-real))


def F_loss(real,fake):
    p0=compute_error(real[0],fake[0])
    p1=compute_error(real[1],fake[1])
    p2=compute_error(real[2],fake[2])
    p3=compute_error(real[3],fake[3])
    p4=compute_error(real[4],fake[4])
    return p0 + p1 + p2 + p3 + p4



def create_block(out_c, in_c, first=True):
    layers = []
    if not first:
        layers.append(nn.MaxPool2d(kernel_size= 2, stride=2, padding=0))
    layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ResUnet(nn.Module):
    def __init__(self, dims=32):
        super(ResUnet, self).__init__()
        
        self.block1 = create_block(dims, 3)
        self.block2 = create_block(dims *2 ,dims, first=False)
        self.block3 = create_block(dims *4 ,dims * 2, first=False)
        self.block4 = create_block(dims *8 ,dims * 4, first=False)
        self.block5 = create_block(dims *16 ,dims * 8, first=False)

        self.up6 = nn.ConvTranspose2d(in_channels=16 * dims, out_channels=8 * dims, kernel_size=2, stride=2, padding=0)
        #after up-sampling, apply torch.cat to 5 and 4
        self.block6 = create_block(in_c= 16 * dims, out_c= 8 * dims)

        self.up7 = nn.ConvTranspose2d(in_channels=8 * dims, out_channels=4 * dims, kernel_size=2, stride=2, padding=0)
        #after up-sampling, apply torch.cat to 6 and 3
        self.block7 = create_block(in_c= 8 * dims, out_c= 4 * dims)

        self.up8 = nn.ConvTranspose2d(in_channels=4 * dims, out_channels=2 * dims, kernel_size=2, stride=2, padding=0)
        #after up-sampling, apply torch.cat to 7 and 2
        self.block8 = create_block(in_c= 4 * dims, out_c= 2 * dims)

        self.up9 = nn.ConvTranspose2d(in_channels=2 * dims, out_channels=dims, kernel_size=2, stride=2, padding=0)
        #after up-sampling, apply torch.cat to 8 and 1
        self.block9 = create_block(in_c= 2 * dims, out_c=dims)

        self.block10 = nn.Conv2d(dims, 3, 1, stride=1, padding=0)

    def forward(self, x):
        #x = x.float()
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x6 = self.up6(x5)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = self.block6(x6)

        x7 = self.up7(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.block7(x7)

        x8 = self.up8(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.block8(x8)

        x9 = self.up9(x8)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.block9(x9)

        x10 = self.block10(x9)
        return x10

class Fcn_resent50(nn.Module):
    def __init__(self, pre_trained = True, num_no_grad = 4) -> None:
        super(Fcn_resent50, self).__init__()
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=pre_trained)
        layers = list(model.backbone)[::-2] # total number of layers is 8
        #turn off gradient for several first layers
        for layer in layers:
            for param in model.backbone[layer].parameters():
                param.require_grad = False
        
        self.model = model
        self.fc_head = nn.Conv2d(21, 3, (1,1), stride=(1,1), padding=(0,0), bias=True) #can switch off bias

    def forward(self, X):
        return self.fc_head(self.model(X)['out'])



# def fcn_resent50(pre_trained=True, num_no_grad = 4):
#     '''
#     Returns a model of FCN resnet 50 (optionally pretrained on COCO)
#     num_no_grad: number of blocks that for which gradient is turned off
#     '''
#     model = torchvision.models.segmentation.fcn_resnet50(pretrained=pre_trained)
#     layers = list(model.backbone)[::-2] # total number of layers is 8
#     #turn off gradient for several first layers
#     for layer in layers:
#         for param in model.backbone[layer].parameters():
#             param.require_grad = False

#     return model
