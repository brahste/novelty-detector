import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
import utils.datasets
import utils.callbacks
import matplotlib.pyplot as plt
import glob

class BinaryCNN(pl.LightningModule):
    
    def __init__(self, params: dict):
        super(BinaryCNN, self).__init__()
        
        self.p = params
        #self.cae = autoencoder
        self.in_channels = int(self.p['in_channels'])
        self.H = self.p['img_size']
        self.loss_function = nn.BCEWithLogitsLoss()
        #self.prepare_data()
        
        #print(type(self.p['hid_dim']))
        
        modules = []
        if self.p['hid_dim'] is None:
            self.hid_dims = [32, 64]
        else:
            self.hid_dims = self.p['hid_dim']
        #self.hid_dims = int(self.p['hid_dim'])
        #print(type(self.hid_dims))
        
        for hid in self.hid_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, hid, kernel_size=5, padding=2),
                    nn.MaxPool2d(2, 2)
                )
            )
            self.in_channels = hid
            self.H /= 2
            #print(self.H)
            #print('New in chans: ', self.in_channels)
            
        self.conv = nn.Sequential(*modules)
        
        modules = []
        
        if self.p['fc_dim'] is None:
            self.fc_dims = [512]
        else:
            self.fc_dims = self.p['fc_dim']
            
        for fc in self.fc_dims:
            #print('num flat chans: ', self.in_channels * int(self.H)**2)
            modules.append(nn.Linear(self.in_channels * int(self.H)**2, fc))
            self.in_channels = fc
            
        self.fc = nn.Sequential(*modules)
        self.output = nn.Linear(self.in_channels, 1)
    
    def forward(self, x):
        #print('type', type(x))
        x = F.relu( self.conv(x) )
        #print('After conv. shape: ', x.shape)
        x = torch.flatten(x, start_dim=1) # Start dim 1 to preserve batches
        #print('After fc shape: ', x.shape)
        x = F.relu( self.fc(x) )
        x = self.output(x)
        return x

    def training_step(self, batch, batch_nb):
        print(batch_nb, 'train')

        
        x, y = batch
        x, y = x.float(), y.float().unsqueeze(1)
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        
        #print(len(x))

        # if batch_nb % 10 == 0:
        #     self.sample_image_from_batch(x, y, y_hat)

        #self.logger.experiment.log({'loss_blah': loss})

        return {
            'loss': loss,
            #'progress_bar': {'train_loss': loss, 'batch_nb': batch_nb},
            'log': {'train_loss': loss, 'batch_nb': batch_nb}
        }

    def training_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {
            'avg_train_loss': avg_val_loss,
            'log': {'avg_train_loss': avg_val_loss}
        }        

    def validation_step(self, batch, batch_nb):
        print(batch_nb, 'val')
        
        x, y = batch
        x, y = x.float(), y.float().unsqueeze(1)
        #print(type(x))
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)

        #print('y_hat', F.softmax(y_hat), 'y' ,y)

        metrics = self.performance_metrics(F.softmax(y_hat), y)
        #self.logger.experiment.log({'metrics': metrics['accuracy']})

        return {
            'val_loss': loss, 'metrics': metrics
            #'progress_bar': {'val_loss': loss, 'batch_nb': batch_nb},
            #'log': {'val_loss': loss, 'metrics': metrics}
        }

    def validation_epoch_end(self, outputs):
        '''outputs is an aray (or I guess.. tensor) of dictionaries, one for each batch'''
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean() 
        #print(outputs)

        avgs = {}
        avgs['avg_val_loss'] = avg_val_loss

        for batch_nb in outputs:
            for metr in batch_nb['metrics']:

                avgs['avg_'+metr] = batch_nb['metrics'][metr]


            # metrics = batch_result['metrics']
            # metrics['avg_val_loss'] = avg_val_loss

            #counts = batch_result['counts']
            #precision = batch_result['precision']

            #print(metrics, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        #self.sample_images()
        return {
            'avg_val_loss': avg_val_loss,
            #'log': {'avg_val_loss': avg_val_loss, 'metrics': metrics}
            'log': avgs
        }

    # def training_epoch_end(self, outputs):
    #     avg_loss = torch.stack()

    
    # def train_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #                     self.train_dataset,
    #                     batch_size = self.p['batch_size'],
    #                     shuffle = True, 
    #                     num_workers = self.p['num_workers']
    #     )
    
    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #                     self.val_dataset,
    #                     batch_size = self.p['batch_size'],
    #                     num_workers = self.p['num_workers']
    #     )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                        self.test_dataset,
                        batch_size = 1,
                        num_workers = self.p['num_workers']
        )
    
    def configure_dataset(self, cae):
        
        dataset = utils.datasets.ErrorMapDataset(
                        cae,
                        root = self.p['data_path'],
                        dirs = [self.p['test_nov_dir'], self.p['test_typ_dir']],
                        transform = self.data_transforms()
        )
        
        _, self.test_dataset = torch.utils.data.random_split(dataset, [660, len(dataset)-660])
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(_, [600, 60])

        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def test_step(self, batch, batch_nb):
        x, y, = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        
        return  {'loss': loss}

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.p['LR'])
        
        # Learn how to build learning rate schedulers
        return opt
    
    def data_transforms(self):
        
        normalize_zero_one = lambda inp, bit_depth=8: inp / (2**bit_depth)
        
        transform = torchvision.transforms.Compose(
                        [torchvision.transforms.ToTensor(),
                         torchvision.transforms.Lambda(normalize_zero_one)]
        )
        return transform

    def sample_image_from_batch(self, x, y, y_hat):

        grab = torch.randint(0, len(x), (1,))

        x = x[grab].squeeze()           # Select a random batch element
        x = x.permute(1, 2, 0).cpu()    # prepare for viewing image viewing
        x = x[:,:,2]                    # Select just red channel
        plt.imshow(x)
        plt.savefig('images/saved.png')

    def performance_metrics(self, y_hat, y):
        '''0 is typical, 1 is novel, also works with boolean arrays'''
        counts = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for quadrant in zip(y_hat, y):
            
            if quadrant == (1, 1): # TP
                counts['TP'] += 1
            elif quadrant == (0, 1): # FN
                counts['FN'] += 1
            elif quadrant == (1, 0): # FP
                counts['FP'] += 1
            elif quadrant == (0, 0): # TN
                counts['TN'] += 1
                
        precision = counts['TP'] / (counts['TP'] + counts['FP'])
        recall    = counts['TP'] / (counts['TP'] + counts['FN'])
        accuracy  = (counts['TP'] + counts['TN']) / len(y)
        f1score   = (2 * precision * recall) / (precision + recall)
        
        return {
            'counts': counts,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1score': f1score
        }