import torch
import torch.nn as nn
from model import *
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval(model):
    #TODO: add data loader for dataset

    epochs = 1000
    lr = 1e-3
    bsize = 32
    wd = 1e-4

    optimizer = torch.optim.Adam(model.paramters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=0, verbose=True)
    loss_f = F_loss

    for epoch in range(epochs):
        train_epoch(model, train_dl, optimizer, loss_f, epoch, epochs)
        
        scheduler.step()



def train_epoch(model, train_dl, optimizer, loss_func, epoch, epochs):
    bar = tqdm(train_dl)
    for x1, x2, gt in bar:
        torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad()
        x1, x2, gt = x1.to(device), x2.to(device), gt.to(device)

         # Run through network
        res1 = model(x1)
        res2 = model(x2)
        # Calculate loss and backprop
        loss = loss_func(res1, gt) + loss_func(res2, gt) + loss_func(res1, res2) #two losses in comparison to ground truth and consistency between themselves

        loss.backward()
        optimizer.step()

        bar.set_description(f'TEpoch:[{epoch+1}/{epochs}]', refresh=True) 

