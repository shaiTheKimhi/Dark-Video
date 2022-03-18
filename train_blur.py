import torch
import torch.nn as nn
from model import *
from tqdm import tqdm
from dataset import *
from dataset_blur import * 
from json import dumps
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval(vgg, unet, loss_f=F_loss, lam=0.05):
    epochs = 100
    lr = 1e-4
    wd = 1e-4
    bs = 2 #batch size

    train_logs = []
    val_logs = []

    ratio = 0.7

    train_ds, train_ds_list = getDataset('train')
    test_ds, test_ds_list =  getDataset('test')
    

    # train_ds, test_ds = create_dataset(train_ratio=ratio) #path is already default
    # train_dl = torch.utils.data.DataLoader(train_ds,batch_size=bs, shuffle=True)
    # test_dl = torch.utils.data.DataLoader(test_ds,batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=0, verbose=True)
    max_psnr = -float('inf')


    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_logs.append(train_epoch(vgg, unet, train_ds, optimizer, loss_f, epoch, epochs, lam).clone().detach().to('cpu'))

       
        torch.cuda.empty_cache()
        t = valid_epoch(vgg, unet, test_ds, loss_f, epoch, epochs, None)
        val_logs.append(t.clone().detach().to('cpu'))
        

        #torch.cuda.empty_cache()
        #val_logs.append(valid_epoch(vgg, unet, test_dl, loss_f, epoch, epochs, None))
        

        scheduler.step()
        
        #loss = min_loss #TODO: remove this line
        psnr = val_logs[-1]
        if (psnr >= max_psnr):
            torch.save({
            'epoch': epoch,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'psnr': psnr,
            'loss': train_logs[-1],
            },  './checkpoint.pt')
           
            max_psnr = psnr

    return train_logs, val_logs
            



'''
Returns the validation loss for each epoch (no accuracy measure, but can add PSNR measure)
'''
def valid_epoch(vgg, unet, test_ds, loss_func, epoch, epochs, optimizer):
    total_loss = 0
    total_psnr = 0
    total_samples = 0

    for blur, gt in test_ds:
        torch.cuda.empty_cache()
        unet.train()
        blur, gt = blur.to(device), gt.to(device)
        n = blur.shape[0]
        total_samples += n

        y1 = unet(blur)
        total_loss += optimize(vgg, y1, gt, optimizer, loss_func)
        total_psnr += psnr(y1, gt)
        # Add evaluation for PSNR and FSNR and return these values
       
        bar.set_description(f'Validation:[{epoch+1}/{epochs}] psnr:{total_psnr/ total_samples}', refresh=True) 
    
    return (total_psnr/ total_samples).to('cpu')

def eval(model, im1, im2,  loss_func):
    res1 = model(im1)
    res2 = model(im2)
    loss = loss_func(res1, res2)
    return loss

def train_epoch(vgg, unet, train_ds, optimizer, loss_func, epoch, epochs, lam):
    total_loss = 0
    total_samples = 0
    for x1, x2, gt in train_ds:
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
        loss = loss_func(t3, t1) + loss_func(t3, t2) + lam * loss_func(t1, t2)
        loss.backward()
        optimizer.step()

        total_loss += loss
        #total_loss += optimize(vgg, y1, gt, optimizer, loss_func)
        #total_loss += optimize(vgg, y2, gt, optimizer, loss_func)
        #total_loss += optimize(vgg, y1, y2, optimizer, loss_func)

       

        #loss.backward()
        #optimizer.step()


        #total_loss += loss

        # bar.set_description(f'TEpoch:[{epoch+1}/{epochs}] loss:{total_loss/ total_samples}', refresh=True) 
    
    return (total_loss / total_samples).to('cpu')

def psnr(im1, im2):
   #max is to be changed
    max = torch.max(im1.max(), im2.max())
    if max < 0:
        max = torch.max(im1.abs().max(), im2.abs().max())
    return 20 * torch.log10(max / (torch.sqrt(torch.mean((im1 - im2) ** 2))))


def optimize(model, im1, im2, optimizer, loss_func):
    res1 = model(im1)
    res2 = model(im2)
    loss = loss_func(res1, res2)

    loss.backward()
    if optimizer:
        optimizer.step()

    return loss


def train_epoch_m(vgg, unet, unet_m, train_dl, optimizer, loss_func, epoch, epochs, m, lam):
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
        y2 = unet_m(x2)

        t1 = vgg(y1)
        t2 = vgg(y2)
        t3 = vgg(gt)
        loss = loss_func(t3, t1) + loss_func(t3, t2) + lam * loss_func(t1, t2)
        loss.backward()
        optimizer.step()

        total_loss += loss

        bar.set_description(f'TEpoch:[{epoch+1}/{epochs}] loss:{total_loss/ total_samples}', refresh=True) 

    #update momentum model params
    enc_params = zip(unet.parameters(), unet_m.parameters())
    for q_parameters, k_parameters in enc_params:
        k_parameters.data = k_parameters.data * m + q_parameters.data * (1. - m)
       
    return (total_loss/total_samples).to('cpu')
        
    

def momentum_train(vgg, unet, lam=0.05, loss_f=F_loss):
    epochs = 100
    lr = 1e-4
    wd = 1e-4
    bs = 2 #batch size
    momentum = 0.999
    import copy
    momentum_encoder = copy.deepcopy(unet).to(device)
    for params in momentum_encoder.parameters(): #Momentum model does not require gradient
        params.requires_grad = False

    ratio = 0.7

    train_ds, test_ds = create_dataset(train_ratio=ratio) #path is already default
    train_dl = torch.utils.data.DataLoader(train_ds,batch_size=bs, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds,batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=0, verbose=True)
    max_psnr = -float('inf')


    train_logs = []
    val_logs = []

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_logs.append(train_epoch_m(vgg, unet, momentum_encoder, train_dl, optimizer, loss_f, epoch, epochs, momentum, lam).clone().detach().to('cpu'))

       
        torch.cuda.empty_cache()
        t = valid_epoch(vgg, unet, test_dl, loss_f, epoch, epochs, None)
        val_logs.append(t.clone().detach().to('cpu'))

        scheduler.step()
        
       
        if (val_logs[-1] >=  max_psnr):
            torch.save({
                    'psnr' : max_psnr,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),'epoch' : epoch, 
                    }, './checkpoint.pt')
            max_psnr = val_logs[-1]

    return train_logs, val_logs



if __name__ == "__main__":
    serial = str(len(os.listdir("./logs")))

    print(device)
    vgg = Vgg19().to(device)
    unet =  ResUnet().to(device)
    logs = train_eval(vgg, unet)

    #train with momentum method (add logs save)
    #logs = momentum_train(vgg, unet)

    #train without special loss (add logs save)
    #logs = train_eval(nn.Identity(), unet, compute_error) #compute error is MSE difference between two images

    #train with transfer learning from COCO (add logs save)
    #fcn_net = Fcn_resent50().to(device)
    #logs = train_eval(vgg, fcn_net)
    #logs = train_eval(nn.Identity(), fcn_net, compute_error)

    logs = torch.tensor(logs).tolist()

    file = open(f"logs/naive{serial}.txt","w")
    file.write(dumps(logs))
    file.close()


