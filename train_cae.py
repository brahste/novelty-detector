"""
Created on Fri May 15 20:52:13 2020

@author: Braden Stefanuk
"""
import argparse, os, yaml
import models.cae
import utils.callbacks
import utils.datasets
import torch
import pytorch_lightning as pl

# Catalog available models
cae_models = {'CAE': models.cae.CAE}
#cnn_models = {'BinaryCNN': models.BinaryCNN}

parser = argparse.ArgumentParser(description='Runs novelty detection related experiments')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/cae.yaml')
args = parser.parse_args()
print(args.filename)

with open(str(args.filename), 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        
tt_logger = pl.loggers.TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)
        
#cae = models.KernerCAE(config['exp_params'])
cae = cae_models[config['exp_info']['model_name']](config['exp_params'])
checkpoint_callback=pl.callbacks.model_checkpoint.ModelCheckpoint(config['logging_params']['save_dir'])
callbacks=[utils.callbacks.SaveStateDictCallback()]

trainer = pl.Trainer(
                default_save_path=f'{tt_logger.save_dir}',
                checkpoint_callback=checkpoint_callback,
                callbacks=callbacks,
                min_nb_epochs=1,
                logger=tt_logger,
                log_save_interval=100,
                # # train_percent_check=1.,
                # # val_percent_check=1.,
                num_sanity_val_steps=5,
                **config['trainer_params']
            )

fit = True
if fit:
    trainer.fit(cae)