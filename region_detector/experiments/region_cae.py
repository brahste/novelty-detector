import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
# import matplotlib.pyplot as plt

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

    # def on_train_epoch_start(self):
    #     random_integer = torch.randint(low=0, high=10, size=())
    #     self._train_epoch_snapshot_step = self.global_step + random_integer

    # def on_validation_epoch_start(self):
    #     random_integer = torch.randint(low=0, high=10, size=())
    #     self._validation_epoch_snapshot_step = self.global_step + random_integer


    def training_step(self, batch, batch_idx):
        print('Training global step: ', self.global_step)

        x_in = batch
        z = self.encoder(batch)
        x_out = self.decoder(z)

        loss = self._loss_function(x_out, x_in)

        result = {
            'x_in': x_in, # Tensor
            'x_out': x_out, # Tensor
            'loss': loss # Scalar
        }

        # # Log some data
        # if self.global_step == self._train_epoch_snapshot_step:
        #     self._handle_logging(result, session='train')

        self.logger.experiment.add_scalar('training_loss', loss, global_step=self.global_step)
        return result

    def validation_step(self, batch, batch_idx):
        print('Validation global step: ', self.global_step)

        x_in = batch
        z = self.encoder(batch)
        x_out = self.decoder(z)
        
        loss = self._loss_function(x_out, x_in)

        result = {
            'x_in': x_in, # Tensor
            'x_out': x_out, # Tensor
            'loss': loss # Scalar
        }

        # Log some data
        if self.global_step == self._validation_epoch_snapshot_step:
            self._handle_logging(result, session='train')

        self.logger.experiment.add_scalar('validation_loss', loss, global_step=self.global_step)
        return result

    def _handle_logging(self, result: dict, session: str='train'):

        compute = {
            'x_in_01': tools.unstandardize_batch(result['x_in']),
            'x_out_01': tools.unstandardize_batch(result['x_out']),
            'error_map': tools.get_error_map(result['x_in'], result['x_out'])
        }

        self._log_to_tensorboard(result, compute)
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
                    f'{self.global_step}-{key}.png'
                )
            )
    
    
