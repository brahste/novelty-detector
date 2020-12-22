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
        self.in_channels = int(self.p['in_channels'])
        self.H = self.p['img_size']
        self.loss_function = nn.BCEWithLogitsLoss()

        modules = []
        if self.p['hid_dim'] is None:
            self.hid_dims = [32, 64]
        else:
            self.hid_dims = self.p['hid_dim']

        for hid in self.hid_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, hid, kernel_size=5, padding=2),
                    nn.LeakyReLU(1e-2),               
                    nn.MaxPool2d(2, 2),
                    nn.BatchNorm2d(hid)
                )
            )
            self.in_channels = hid
            self.H /= 2
            
        # Construct convolutional section of network
        self.conv = nn.Sequential(*modules)
        
        modules = []
        if self.p['fc_dim'] is None:
            self.fc_dims = [512]
        else:
            self.fc_dims = self.p['fc_dim']
            
        for fc in self.fc_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(self.in_channels * int(self.H)**2, fc),
                    nn.LeakyReLU(1e-2),
                    nn.BatchNorm1d(fc)
                )
            )
            self.in_channels = fc
            
        # Construct fully connected section of network
        self.fc = nn.Sequential(*modules)
        self.output = nn.Linear(self.in_channels, 1)
    
    def forward(self, x):
        x = F.relu( self.conv(x) )
        x = torch.flatten(x, start_dim=1)  # dim 1 preserves batches
        x = F.relu( self.fc(x) )
        return self.output(x)


    def training_step(self, batch, batch_nb):
        x, y = batch
        x, y = x.float(), y.float().unsqueeze(1)
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        return {
            'loss': loss,
            'log': {'train_loss': loss, 'batch_nb': batch_nb}
        }

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {
            'avg_train_loss': avg_train_loss,
            'log': {'avg_train_loss': avg_train_loss}
        }        

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x, y = x.float(), y.float().unsqueeze(1)
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)

        # Grab performance metrics with set threshold
        metrics = self.performance_metrics(torch.sigmoid(y_hat), y, 0.5)
        return {
            'val_loss': loss, 'metrics': metrics
        }

    def validation_epoch_end(self, outputs):

        self.visualize_results()

        logs = self.handle_metrics(outputs)
        # Handle remaining metrics
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean() 
        logs['avg_val_loss'] = avg_val_loss
        return {
            'avg_val_loss': avg_val_loss,
            'log': logs
        }

    def test_step(self, batch, batch_nb):
        x, y = batch
        x, y = x.float(), y.float().unsqueeze(1)
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)

        metrics = self.performance_metrics(torch.sigmoid(y_hat), y, 0.5)
        return {
            'test_loss': loss, 'metrics': metrics
        }

    def test_epoch_end(self, outputs):

        self.visualize_results()

        logs = self.handle_metrics(outputs)        
        # Handle remain metrics
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean() 
        logs['avg_test_loss'] = avg_test_loss
        return {
            'avg_test_loss': avg_test_loss,
            'log': logs
        }

    def handle_metrics(self, outputs):
        logs = {}
        counts = {}
        # To keep track of which batches return 'Metrics calc failed'
        # assign the variable less, which is used to remove the
        # number of failed batches from the divisor in when calculating
        # the mean.
        less = 0

        for val_batch in outputs:
            v = val_batch['metrics']
            if v == 'Metrics calculation failed':
                less += 1
            else:
                for metr in v:
                    if metr == 'counts':
                        for q in v[metr]:
                            if q in counts:
                                counts[q] += v[metr][q]
                            else:
                                counts[q] = v[metr][q]
                    else:
                        if 'avg_'+metr in logs:
                            logs['avg_'+metr] += v[metr]
                        else:
                            logs['avg_'+metr] = v[metr]
        logs = {key: logs[key] / (len(outputs) - less) for key in set(logs)}
        # Add counts dictionary to log output
        logs['counts'] = counts
        return logs
    
    def configure_dataset(self, cae):
        '''Configures the Error Map dataset, function is called in 
        'train_cnn.py' before declaring the dataloaders
        '''
        dataset = utils.datasets.ErrorMapDataset(
                        cae,
                        root = self.p['data_path'],
                        dirs = [self.p['test_typ_dir'], self.p['test_nov_dir']],
                        transform = self.data_transforms()
        )
        # Consider building a dynamic dataset splitter in future
        _, self.test_dataset = torch.utils.data.random_split(dataset, [660, len(dataset)-660])
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(_, [600, 60])

        return self.train_dataset, self.val_dataset, self.test_dataset

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.p['LR'])
        # Learn how to build learning rate schedulers...
        return opt
    
    def data_transforms(self):
        '''Composition of transforms used when configuring the dataset
        '''
        normalize_zero_one = lambda inp, bit_depth=8: inp / (2**bit_depth)
        
        return torchvision.transforms.Compose(
                        [torchvision.transforms.ToTensor(),
                         torchvision.transforms.Lambda(normalize_zero_one)]
        )

    def performance_metrics(self, y_hat, y, threshold):
        
        # Threshold and convert to boolean arrays
        novelties = (y_hat > threshold).to(torch.bool)
        labels    = y.to(torch.bool)
        
        # Convention: 0 -> typical, 1 -> novel
        counts = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for quadrant in zip(novelties, labels):
            
            if quadrant   == (1, 1): # TP
                counts['TP'] += 1
            elif quadrant == (0, 1): # FN
                counts['FN'] += 1
            elif quadrant == (1, 0): # FP
                counts['FP'] += 1
            elif quadrant == (0, 0): # TN
                counts['TN'] += 1

        # Early in training sometimes poor performance raises divide by
        # zero error.
        try:   
            precision = counts['TP'] / (counts['TP'] + counts['FP'])
            recall    = counts['TP'] / (counts['TP'] + counts['FN'])
            accuracy  = (counts['TP'] + counts['TN']) / len(y)
            f1score   = (2 * precision * recall) / (precision + recall)
            
            return {
                'counts':    counts,
                'precision': torch.tensor(precision),
                'recall':    torch.tensor(recall),
                'accuracy':  torch.tensor(accuracy),
                'f1score':   torch.tensor(f1score)
            }
        except:
            return 'Metrics calculation failed'

    def visualize_results(self):
        # Get sample reconstruction image
        x, y = next(iter(self.val_dataloader()))
        x = x.to('cuda').float()
        y = y.to('cuda').float()
        y_hat = self.forward(x)

        # Pick random batch elements to sample
        pick = torch.randint(0, len(x), (4,))

        fig, ax = plt.subplots(1, 4)
        for i in range(4):
            emap, y_pick, y_hat_pick = self.clean_batch_for_display(pick[i], x, y, y_hat)

            ax[i].imshow(emap[...,2])
            ax[i].text(0, 0, 
                       f'Actual: {y_pick:.2f}, Predicted: {y_hat_pick:.2f}',
                       fontsize=6)
        plt.savefig(f'{self.logger.save_dir}{self.logger.name}/'
                    + f'version_{self.logger.version}/'
                    + f'CnnOutputVis_E{self.current_epoch}.png')
        del fig

    def clean_batch_for_display(self, b: int, x, y, y_hat):
        '''For casting batchs to the cpu and grabing the specified element
        from the batch
        '''
        emap = x[b].cpu()
        y_pick = y.view(-1)[b].cpu()
        
        y_hat = torch.sigmoid(y_hat)
        y_hat_pick = y_hat.view(-1)[b].cpu()
        
        return emap.permute(1, 2, 0), y_pick.item(), y_hat_pick.item()