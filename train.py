import torch
import torch.nn as nn
from model import *
from tqdm import tqdm
from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval(vgg, unet):
    epochs = 100
    lr = 3e-4
    wd = 1e-4
    bs = 2 #batch size

    ratio = 0.7

    train_ds, test_ds = create_dataset(train_ratio=ratio) #path is already default
    train_dl = torch.utils.data.DataLoader(train_ds,batch_size=bs, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds,batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=0, verbose=True)
    loss_f = F_loss
    min_loss = float('inf')


    for epoch in range(epochs):
        train_epoch(vgg, unet, train_dl, optimizer, loss_f, epoch, epochs)

        #loss = valid_epoch(model, test_dl, loss_f, epoch, epochs, None)

        scheduler.step()
        
        loss = min_loss #TODO: remove this line
        if (loss <= min_loss):
            torch.save({
                    'validation_loss' : loss,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, './checkpoint.pt')
            min_loss = loss
            



'''
Returns the validation loss for each epoch (no accuracy measure, but can add PSNR measure)
'''
def valid_epoch(vgg, unet, train_dl, loss_func, epoch, epochs, optimizer):
    total_loss = 0
    total_samples = 0
    bar = tqdm.tqdm(train_dl)
    for x1, x2, gt in bar:
        torch.cuda.empty_cache()
        model.train()
        x1, gt = x1.to(device), gt.to(device)
        n = x1.shape[0]
        total_samples += n

        total_loss += optimize(model, x1, gt, optimizer, loss_func)
       
        bar.set_description(f'Validation:[{epoch+1}/{epochs}] loss:{total_loss/ total_samples}', refresh=True) 
    
    return total_loss

def eval(model, im1, im2,  loss_func):
    res1 = model(im1)
    res2 = model(im2)
    loss = loss_func(res1, res2)
    return loss

def train_epoch(vgg, unet, train_dl, optimizer, loss_func, epoch, epochs):
    total_loss = 0
    total_samples = 0
    bar = tqdm.tqdm(train_dl)
    for x1, x2, gt in bar:
        torch.cuda.empty_cache()
        unet.train()
        optimizer.zero_grad()
        x1, x2, gt = x1.to(device), x2.to(device), gt.to(device)
        n = x1.shape[0]
        total_samples += n

        y1 = unet(x1)
        y2 = unet(x2)

        t1 = vgg(y1)
        t2 = vgg(y2)
        t3 = vgg(gt)
        loss = loss_func(t1, t3) + loss_func(t2, t3) + loss_func(t1, t2)
        loss.backward()
        optimizer.step()

        total_loss += loss
        #total_loss += optimize(vgg, y1, gt, optimizer, loss_func)
        #total_loss += optimize(vgg, y2, gt, optimizer, loss_func)
        #total_loss += optimize(vgg, y1, y2, optimizer, loss_func)

       

        #loss.backward()
        #optimizer.step()


        #total_loss += loss

        bar.set_description(f'TEpoch:[{epoch+1}/{epochs}] loss:{total_loss/ total_samples}', refresh=True) 


def optimize(model, im1, im2, optimizer, loss_func):
    res1 = model(im1)
    res2 = model(im2)
    loss = loss_func(res1, res2)

    loss.backward()
    if optimizer:
        optimizer.step()

    return loss




if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    print(device)
    vgg = Vgg19().to(device)
    unet =  ResUnet().to(device)
    train_eval(vgg, unet)
