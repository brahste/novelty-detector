import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import utils.datasets
import matplotlib.pyplot as plt

class CAE(pl.LightningModule):

    def __init__(self,
                 params: dict) -> None:
        super(CAE, self).__init__()

        self.p = params
        self.H = self.p['img_size']
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hold_graph = False
        self.loss_function = nn.MSELoss()
        
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

    def encode(self, x):
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
        return {
            'loss': loss,
            'log':  {'train_loss': loss, 'batch_nb': batch_nb}
        }        

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x, y = x.float(), y.float()
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return {
            'val_loss': loss
        }

    def validation_epoch_end(self, outputs):
        self.sample_images()

        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {
            'val_loss': avg_val_loss, 
            'log': {'avg_val_loss': avg_val_loss}
        }
    
    def test_step(self, batch, batch_nb):
        x, y, = batch
        x, y = x.float(), y.float()
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return  {
            'loss': loss
        }

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.p['LR'])
        # Learn how to build learning rate schedulers
        return opt

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.p['dataset'] == 'Mastcam':
            dataset = utils.datasets.MastcamDataset(
                            path = os.path.join(self.p['data_path'], self.p['tng_dir']),
                            transform = self.data_transforms(),
                            label = torch.tensor(0)
            )
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return torch.utils.data.DataLoader(
                        dataset, 
                        batch_size = self.p['batch_size'], 
                        shuffle = True, 
                        drop_last = True,
                        num_workers = self.p['num_workers']
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:

        if self.p['dataset'] == 'Mastcam':
            dataset = utils.datasets.MastcamDataset(
                            path = os.path.join(self.p['data_path'], self.p['val_dir']),
                            transform = self.data_transforms(),
                            label = torch.tensor(1)
            )
            self.num_val_imgs = len(dataset)
        else:
            raise ValueError('Undefined dataset type')
    
        return torch.utils.data.DataLoader(
                        dataset, 
                        batch_size = self.p['batch_size'], 
                        drop_last = True,
                        num_workers = self.p['num_workers']
        )

    def test_dataloader(self):
        pass

    def data_transforms(self):
        
        normalize_zero_one = lambda inp, bit_depth=8: inp / (2**bit_depth)
        
        return torchvision.transforms.Compose(
                        [torchvision.transforms.ToTensor(),
                         torchvision.transforms.Lambda(normalize_zero_one)]
        )

    def sample_images(self):
        # Get sample reconstruction image
        x, y = next(iter(self.val_dataloader()))
        x = x.to('cuda').float()
        y = y.to('cuda').float()
        x_hat = self.forward(x)
        error_map = self.squared_error(x_hat, x)

        select = [3, 33]

        fig, ax = plt.subplots(2, 8, figsize=(20,10))
        for view in range(2):
            emap = self.clean_batch_for_display(select[view], x, x_hat, error_map)

            for c in range(2,8):
            # plt.sca(ax[i])
                ax[view,c].imshow(emap[...,c-2])
            #ax[i].set_title('batch number ', random_idx[i])
            ax[view,0].imshow(x[select[view],2].cpu())
            ax[view,1].imshow(x_hat[select[view],2].cpu())
        plt.savefig(f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/ValErrorMap_E{self.current_epoch}.png")
        del fig

    def clean_batch_for_display(self, b: int, x, x_hat, error_map):

        emap = error_map[b].cpu()

        maxs = torch.max(torch.max(emap, dim=1)[0], dim=1)[0]
        maxs = maxs[...,None,None]

        # Error map
        norm_emap = emap / maxs
        return norm_emap.permute(1, 2, 0)