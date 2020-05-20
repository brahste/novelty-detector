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
	cae.load_state_dict(torch.load('logs/CAE_.pt',map_location=device))
	print('cae device', cae)

cnn = models.cnn.BinaryCNN(config['exp_params'])
print('cnn device ', cnn)

# Declare logger
logger = pl.loggers.TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

train, val, test = cnn.configure_dataset(cae)
train_loader = torch.utils.data.DataLoader(
                        train,
                        batch_size = config['exp_params']['batch_size'],
                        shuffle = True, 
                        num_workers = config['exp_params']['num_workers']
)

val_loader = torch.utils.data.DataLoader(
                        val,
                        batch_size = config['exp_params']['batch_size'],
                        num_workers = config['exp_params']['num_workers']
)

trainer = pl.Trainer(default_save_path=f'{logger.save_dir}',
                	# checkpoint_callback=checkpoint_callback,
                	# callbacks=callback,
					min_nb_epochs=1,
                	logger=logger,
                	log_save_interval=5,
	                num_sanity_val_steps=5,
	                **config['trainer_params'])

trainer.fit(cnn, train_loader, val_loader)
