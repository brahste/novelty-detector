import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from utils import tools
from utils.datasets.lunar_analogue import LunarAnalogueDataModule


class RegionCAE(pl.LightningModule):
    '''
    The lighting module helps enforce best practices
    by keeping your code modular and abstracting the
    'engineering code' or boilerplate that new model
    require.
    '''
    def __init__(
            self, 
            params: dict
        ):
        super(RegionCAE, self).__init__()

        self.dm = LunarAnalogueDataModule(params)
        self.dm.prepare_data()
        self.dm.setup()

        # Anything set as an hparam is automatically saved, so it's a convenient
        # way to store and use hyperparameters
        self.hparams = params
        if self.hparams.learning_rate is None:
            self.learning_rate = float(0.)
        print(f'Initializing with parameters:\n{self.hparams}\n')

        # Return a callable torch.nn.XLoss object
        ##### e.g. self._loss_function = self._handle_loss_function(params['loss_function'])
        self._loss_function = nn.MSELoss()

        # Encoding layers
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, stride=2)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1, stride=2)
        
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoding layers
        self.conv6_3 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.conv6_2 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.conv6_1 = nn.ConvTranspose2d(512, 512, 3, padding=1, stride=2, output_padding=1)
        self.conv7_3 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.conv7_2 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.conv7_1 = nn.ConvTranspose2d(512, 256, 3, padding=1, stride=2, output_padding=1)
        self.conv8_3 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv8_2 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv8_1 = nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2, output_padding=1)
        self.conv9_2 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.conv9_1 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1)
        self.conv10_2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.conv10_1 = nn.ConvTranspose2d(64, 3, 3, padding=1, stride=2, output_padding=1)

        # Unpooling layers with bilinear interpolation
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Batch normalization layer
        self.bn3 = nn.BatchNorm2d(3)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)

    def forward(self, x):
        # Simple encoding into latent representation
        # and decoding back to input space
        z_latent = self.encoder(x)
        x_recons = self.decoder(z_latent)
        return x_recons

    def encoder(self, x):
        # print(torch.max(x), torch.min(x), x.size())
        x = F.relu( self.conv1_1(x) )
        x = F.relu( self.conv1_2(x) )
        x = self.bn64(x)
        # x = self.pool(x)
        x = F.relu( self.conv2_1(x) )
        x = F.relu( self.conv2_2(x) )
        x = self.bn128(x)
        # x = self.pool(x)
        x = F.relu( self.conv3_1(x) )
        x = F.relu( self.conv3_2(x) )
        x = F.relu( self.conv3_3(x) )
        x = self.bn256(x)
        # x = self.pool(x)
        x = F.relu( self.conv4_1(x) )
        x = F.relu( self.conv4_2(x) )
        x = F.relu( self.conv4_3(x) )
        x = self.bn512(x)
        # x = self.pool(x)
        x = F.relu( self.conv5_1(x) )
        x = F.relu( self.conv5_2(x) )
        x = F.relu( self.conv5_3(x) )
        x = self.bn512(x)
        # x = self.pool(x)
        z_latent = x
        # print(torch.max(x), torch.min(x), x.size())
        return z_latent
    
    def decoder(self, z):
        # print(torch.max(z), torch.min(z), z.size())
        z = F.relu( self.conv6_3(z) )
        z = F.relu( self.conv6_2(z) )
        z = F.relu( self.conv6_1(z) )
        z = self.bn512(z)
        # z = self.unpool(z)
        z = F.relu( self.conv7_3(z) )
        z = F.relu( self.conv7_2(z) )
        z = F.relu( self.conv7_1(z) )
        z = self.bn256(z)
        # z = self.unpool(z)
        z = F.relu( self.conv8_3(z) )
        z = F.relu( self.conv8_2(z) )
        z = F.relu( self.conv8_1(z) )
        z = self.bn128(z)
        # z = self.unpool(z)
        z = F.relu( self.conv9_2(z) )
        z = F.relu( self.conv9_1(z) )
        z = self.bn64(z)
        # z = self.unpool(z)
        z = F.relu( self.conv10_2(z) )
        z = F.relu( self.conv10_1(z) )
        z = self.bn3(z)
        # z = self.unpool(z)
        x_recons = z
        # print(torch.max(x_recons), torch.min(x_recons), x_recons.size())
        return x_recons

    def train_dataloader(self):
        return self.dm.train_dataloader()

    def val_dataloader(self):
        return self.dm.val_dataloader()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate
        )

    def on_epoch_start(self):
        random_integers = torch.randint(
            low=0, 
            high=self.hparams.num_train_batches,
            size=(3,)
        )
        self._random_train_steps = self.global_step + random_integers
        print(f'\nLogging images from batches: {self._random_train_steps.tolist()}\n')

    def training_step(self, batch, batch_idx):

        x_in = batch
        z = self.encoder(batch)
        x_out = self.decoder(z)

        loss = self._loss_function(x_out, x_in)

        images = {
            'x_in': x_in.detach(), # Tensor
            'x_out': x_out.detach() # Tensor
        }
        result = {
            'loss': loss
        }

        # Log some data
        if any([x == self.global_step for x in self._random_train_steps]):
            self._handle_image_logging(images, session='train')

        self.log_dict(result)
        return result # The returned object must contain a 'loss' key

    def validation_step(self, batch, batch_idx):

        x_in = batch
        z = self.encoder(batch)
        x_out = self.decoder(z)
        
        loss = self._loss_function(x_out, x_in)

        result = {
            'val_loss': loss
        }

        self.log_dict(result)
        return result

    def _handle_image_logging(self, images: dict, session: str='train'):

        if self.logger.version is None:
            return
        else:
            compute = {
                'x_in_01': tools.unstandardize_batch(images['x_in']),
                'x_out_01': tools.unstandardize_batch(images['x_out']),
                'error_map': tools.get_error_map(images['x_in'], images['x_out'])
            }

            self._log_to_tensorboard(images, compute)
            self._log_images(compute)
        return


    def _log_to_tensorboard(self, result: dict, compute: dict):
        self.logger.experiment.add_image(
            f'x_in-{self.global_step}', 
            result['x_in'], 
            global_step=self.global_step, 
            dataformats='NCHW'
        )
        self.logger.experiment.add_image(
            f'x_in_unstandardized-{self.global_step}', 
            compute['x_in_01'], 
            global_step=self.global_step, 
            dataformats='NCHW'
        )
        self.logger.experiment.add_image(
            f'x_out-{self.global_step}', 
            result['x_out'], 
            global_step=self.global_step, 
            dataformats='NCHW'
        )
        self.logger.experiment.add_image(
            f'x_out_unstandardized-{self.global_step}', 
            compute['x_out_01'], 
            global_step=self.global_step, 
            dataformats='NCHW'
        )
        self.logger.experiment.add_image(
            f'metric_squared_error-{self.global_step}', 
            compute['error_map'], 
            global_step=self.global_step, 
            dataformats='NCHW'
        )
    
    def _log_images(self, compute: dict):

        logger_save_path = os.path.join(
            self.logger.save_dir, 
            self.logger.name, 
            f'version_{self.logger.version}'
        )

        if not os.path.exists(os.path.join(logger_save_path, 'images')):
            os.mkdir(os.path.join(logger_save_path, 'images'))

        rint = torch.randint(0, len(compute['x_in_01']), size=())
        for key in compute:
            torchvision.utils.save_image(
                compute[key][rint], 
                os.path.join(
                    logger_save_path,
                    'images',
                    f'{self.current_epoch}-{self.global_step-(self.hparams.num_train_batches*self.current_epoch)}-{key}.png'
                )
            )
