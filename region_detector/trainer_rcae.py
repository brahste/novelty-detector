import sys
sys.path.append('.')

import pytorch_lightning as pl
import torch

from experiments.region_cae import RegionCAE
from utils.datasets.lunar_analogue import LunarAnalogueDataModule

import os
print(os.getcwd())

# Eventually these will be imported from a configuration file (.yaml)
params = {
    'experiment_name': 'RCAE',
    'log_directory': 'region_detector/logs',
    'root_data_path': '/home/brahste/Datasets/LunarAnalogue/images-screened',
    'hparams': {
        'train_fraction': 0.75,
        'learning_rate': 0.005,
        'batch_size': 8
    }
}

datamodule = LunarAnalogueDataModule(params)
datamodule.prepare_data()
datamodule.setup('fit')

# Add information about the dataset to the experimental parameters
params['num_train_samples'] = datamodule.num_train_samples
params['num_val_samples'] = datamodule.num_val_samples

autoencoder = RegionCAE(params)

logger = pl.loggers.TensorBoardLogger(
    params['log_directory'], 
    name=params['experiment_name']
)

trainer = pl.Trainer(
    gpus=1,
    logger=logger,
    max_epochs=3
)

trainer.fit(autoencoder, datamodule)
