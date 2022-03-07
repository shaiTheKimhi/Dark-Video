import torch

d = torch.load('checkpoint.pt')
print(d['validation_loss'])
