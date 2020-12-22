#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:49:37 2020

@author: brahste
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import models.cae
import models.cnn
import argparse
import yaml, os
import utils.datasets
import utils.callbacks
import torchvision
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Runs novelty detection related experiments')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/cnn.yaml')
args = parser.parse_args()

with open(str(args.filename), 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

device = torch.device('cpu')
with torch.no_grad():
	cae = models.cae.CAE(config['exp_params'])
	cae.load_state_dict(torch.load('logs/CAE_.pt', map_location=device))
	print('cae device', cae)

cnn = models.cnn.BinaryCNN(config['exp_params'])

train_set, val_set, test_set = cnn.configure_dataset(cae)

q, p = next(iter(train_set))

q = q.permute(1, 2, 0)

plt.imshow(q[...,0])
plt.show()
print(q[...,0].min(), q[...,0].max())


#%%

train_loader = torch.utils.data.DataLoader(
                        train_set,
                        batch_size = config['exp_params']['batch_size'],
                        shuffle = True, 
                        num_workers = config['exp_params']['num_workers']
)

#%%

x, y = next(iter(train_loader))
#%%
emap = x[0].cpu()
print(torch.unique(emap))

y = y.view(-1)[0].cpu()
#y_hat = y_hat.view(-1)[b].cpu()
maxs = torch.max(torch.max(emap, dim=1)[0], dim=1)[0]
maxs = maxs[...,None,None] # Equivalent to double unsqueeze
norm_emap = (emap / maxs).permute(1, 2, 0)

plt.imshow(norm_emap[...,0])