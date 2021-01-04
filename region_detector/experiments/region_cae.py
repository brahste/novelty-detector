import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from utils import tools


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

        self.save_hyperparameters(params)
        self._p = params
        # self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Initializing with device: {self.device}')
        self._num_train_batches = int(self._p['num_train_samples']/self._p['hparams']['batch_size'])

        # Returns a callable torch.nn.XLoss object
        ##### e.g. self._loss_function = self._handle_loss_function(params['loss_function'])
        self._loss_function = nn.MSELoss()

        # Encoding layers
        self.conv1 = nn.Conv2d(3, 9, 5, padding=2)
        self.conv2 = nn.Conv2d(9, 12, 5, padding=2)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoding layers
        self.conv3 = nn.Conv2d(12,  9, 5, padding=2)
        self.conv4 = nn.Conv2d(9, 3, 5, padding=2)
        
        # Unpooling layers with bilinear interpolation
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Simple encoding into latent representation
        # and decoding back to input space
        z_latent = self.encoder(x)
        x_recons = self.decoder(z_latent)
        return x_recons

    def encoder(self, x):
        x = F.relu( self.conv1(x) )
        x = self.pool(x)
        z_latent = F.relu( self.conv2(x) )
        return z_latent
    
    def decoder(self, z):
        z = F.relu( self.conv3(z) )
        z = self.unpool(z)
        x_recons = F.relu( self.conv4(z) )
        return x_recons

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self._p['hparams']['learning_rate']
        )

    def on_train_epoch_start(self):
        random_integers = torch.randint(
            low=0, 
            high=self._num_train_batches,
            size=(3,)
        )
        self._random_train_steps = self.global_step + random_integers
        print(f'EPOCH {self.current_epoch}' + 
            f'Logging images from batches {self._random_train_steps.tolist()}...\n')

    # def on_validation_epoch_start(self):
    #     random_integer = torch.randint(low=0, high=10, size=())
    #     self._validation_epoch_snapshot_step = self.global_step + random_integer


    def training_step(self, batch, batch_idx):
        print('Training global step: ', self.global_step)

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
        # if self.global_step == any(self._random_train_steps):
            self._handle_image_logging(images, session='train')

        self.log_dict(result)
        return result # The returned object must contain a 'loss' key

    def validation_step(self, batch, batch_idx):
        print('Validation global step: ', self.global_step)

        x_in = batch
        z = self.encoder(batch)
        x_out = self.decoder(z)
        
        loss = self._loss_function(x_out, x_in)

        result = {
            'val_loss': loss
        }

        self.log_dict(result)
        return result

    # def validation_epoch_end(self, validation_step_results):
    #     for result in validation_step_results:
    #         running_loss += 


    def _handle_image_logging(self, images: dict, session: str='train'):

        compute = {
            'x_in_01': tools.unstandardize_batch(images['x_in']),
            'x_out_01': tools.unstandardize_batch(images['x_out']),
            'error_map': tools.get_error_map(images['x_in'], images['x_out'])
        }

        self._log_to_tensorboard(images, compute)
        self._log_images(compute)


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
                    f'{self.current_epoch}-{self.global_step-(self._num_train_batches*self.current_epoch)}-{key}.png'
                )
            )
    
    
