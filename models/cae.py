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

class CAE(pl.LightningModule):

    def __init__(self,
                 params: dict) -> None:
        super(CAE, self).__init__()

        self.p = params
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hold_graph = False
        
        # Encoding layers
        self.conv1 = nn.Conv2d(6, 12, 7, padding=3)
        self.conv2 = nn.Conv2d(12, 8, 5, padding=2)
        self.conv3 = nn.Conv2d(8,  3, 3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoding layers
        self.conv4 = nn.Conv2d(3,  8, 3, padding=1)
        self.conv5 = nn.Conv2d(8, 12, 5, padding=2)
        self.conv6 = nn.Conv2d(12, 6, 7, padding=3)
        
        # Unpooling layers with bilinear interpolation
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        try:
            self.hold_graph = self.p['retain_first_backpass']
        except:
            pass

    def encode(self, x):
        #print('encode', x.device)

        x = F.relu( self.conv1(x) )         # 64.64.6  -> 64.64.12
        x = self.pool(x)                    # 64.64.12 -> 32.32.12
        x = F.relu( self.conv2(x) )         # 32.32.12 -> 32.32.8
        x = self.pool(x)                    # 32.32.8  -> 16.16.8
        x = F.relu( self.conv3(x) )         # 16.16.8  -> 16.16.3
        return x
    
    def decode(self, z):
        z = F.relu( self.conv4(z) )         # 16.16.3  -> 16.16.8
        z = self.unpool(z)                  # 16.16.8  -> 32.32.8
        z = F.relu( self.conv5(z) )         # 32.32.8  -> 32.32.12
        z = self.unpool(z)                  # 32.32.12 -> 64.64.12
        z = torch.sigmoid( self.conv6(z) )  # 64.64.12 -> 64.64.6
        return z

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def loss_function(self, x_hat, x):
        return F.mse_loss(x_hat, x)
    
    def squared_error(self, x_hat, x):
        return (x_hat - x)**2

    def training_step(self, batch, batch_nb):
        x, y = batch
        x, y = x.float(), y.float()
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)

        # Error map calculation
        emap = self.squared_error(x_hat, x)
        
        # if batch_nb % 50 == 0:
        #     emap = emap[0].detach().cpu()
        #     emap = emap.squeeze()
        #     print(emap.shape)
        #     emap = emap.permute(1, 2, 0)
        #     plt.imshow(emap[:,:,2])
        #     plt.show()
            
        #     x = x[0].detach().cpu().squeeze()
        #     x = x.permute(1, 2, 0)
        #     plt.imshow(x[:,:,2])
        #     plt.show()
            
        #     x_hat = x_hat[0].detach().cpu().squeeze()
        #     x_hat = x_hat.permute(1, 2, 0)
        #     plt.imshow(x_hat[:,:,2])
        #     plt.show()

        output = {
            'loss': loss,
            'progress_bar': {'train_loss': loss, 'batch_nb': batch_nb},
            'log':          {'train_loss': loss}
        }        
        return output

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x, y = x.float(), y.float()
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        
        output = {
            'val_loss': loss
        }
        return output

    def validation_end(self, outputs):

        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_val_loss}
        #self.sample_images()
        return {'val_loss': avg_val_loss, 'log': tensorboard_logs}
    
    def validation_epoch_end(self, outputs):
        pass
    
    def test_step(self, batch, batch_nb):
        x, y, = batch
        x, y = x.float(), y.float()
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        
        return  {'loss': loss}

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.p['LR'])
        
        # Learn how to build learning rate schedulers
        return opt

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.p['dataset'] == 'Mastcam':
            dataset = utils.datasets.MastcamDataset(
                            path = os.path.join(self.p['data_path'], self.p['tng_dir']),
                            transform = self.data_transforms(),
                            label = torch.tensor([1,0])
            )
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        loader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size = self.p['batch_size'], 
                        shuffle = True, 
                        drop_last = True,
                        num_workers = self.p['num_workers']
        )
        return loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:

        if self.p['dataset'] == 'Mastcam':
            dataset = utils.datasets.MastcamDataset(
                            path = os.path.join(self.p['data_path'], self.p['val_dir']),
                            transform = self.data_transforms(),
                            label = torch.tensor([1,0])
            )
            self.num_val_imgs = len(dataset)
        else:
            raise ValueError('Undefined dataset type')
    
        loader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size = self.p['batch_size'], 
                        drop_last = True,
                        num_workers = self.p['num_workers']
        )
        return loader
    def test_dataloader(self):
        pass

    def data_transforms(self):
        
        normalize_zero_one = lambda inp, bit_depth=8: inp / (2**bit_depth)
        
        transform = torchvision.transforms.Compose(
                        [torchvision.transforms.ToTensor(),
                         torchvision.transforms.Lambda(normalize_zero_one)]
        )
        return transform
    
    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass
        del test_input, recons #, samples