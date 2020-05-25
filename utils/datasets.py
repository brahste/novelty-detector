# Datasets used in Novelty detection experiments
# Author: Braden Stefanuk
# Created: May 4, 2020

import torch, os
import numpy as np
from typing import List

def squared_error(x_hat, x):
    return (x_hat - x)**2

class MastcamDataset(torch.utils.data.Dataset):
    '''Sub-classes dataset class for Mastcam images'''
    def __init__(self, path: str, transform=None, label: torch.tensor=None):
        super(MastcamDataset, self).__init__()
        
        self.path = path
        self.img_files = os.listdir(path) 
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, index):
        
        # Grab image filename and load
        img_file = os.path.join(self.path, self.img_files[index])
        img = np.load(img_file)

        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Handle label
        if self.label is not None:
            return img, self.label
        else:
            return img
        
class ErrorMapDataset(torch.utils.data.Dataset):
        
    def __init__(self, model, root: str=None, dirs: List=None, 
                 transform: str=None, label: torch.tensor=None):
        super(ErrorMapDataset, self).__init__()
        
        self.root = root
        self.dirs = dirs
        self.transform = transform
        self.collect_sets()
        self.model = model
        # self.model.freeze()
        self.model.eval()
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        
        label = self.img_files[index][1]
        img_file = os.path.join(self.root, self.img_files[index][0])
        x = np.load(img_file)
        
        with torch.no_grad():

            if self.transform:
                x = self.transform(x) 
            x = x.float()
            error_map = self.get_error_map(self.model, x)
            return error_map.squeeze(), label
            
    def collect_sets(self):
        self.img_files = []
        for i, path in enumerate(self.dirs):
            for f in os.listdir(os.path.join(self.root, path)):
                file = os.path.join(path, f)
                self.img_files.append( (file, i) )
                
    def to_onehot(self, class_num):
        label = torch.zeros(2)
        label[class_num] = 1
        return label
    
    def get_error_map(self, model, x):
        
        #model.to(torch.device('cpu'))
        x_hat = model(x.unsqueeze(0))
        error_map = squared_error(x_hat, x)
        error_map = self.normalize_error_map(error_map)
        
        return error_map.squeeze()
    
    def normalize_error_map(self, error_map):

        maxs = torch.max(torch.max(error_map, dim=2)[0], dim=2)[0]
        maxs = maxs[...,None,None]


        # print(error_map[b].shape)

        # maxs, max_idxs = torch.max(error_map[b], dim=1)
        # maxs, max_idxs = torch.max(maxs, dim=1)
        # print(maxs.shape)

        # Error map
        normalized = error_map / maxs

        return normalized

        # return emap.permute(1, 2, 0)