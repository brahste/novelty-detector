import sys
import yaml
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner

sys.path.append('.')
from experiments.region_cae import RegionCAE
from utils.datasets.lunar_analogue import LunarAnalogueDataModule

# Set defaults
DEFAULT_CONFIG_FILE = 'region_detector/config/program-config.yaml'


def handle_command_line_arguments():
    if len(sys.argv) == 1:
        return DEFAULT_CONFIG_FILE
    elif len(sys.argv) == 2:
        return sys.argv[0]
    else:
        raise ValueError('More than one configuration file was provided.')


def main():
    config_file = handle_command_line_arguments()

    with open(config_file) as f:
        config = yaml.full_load(f)

    # Initialize datamodule
    datamodule = LunarAnalogueDataModule(config)
    datamodule.prepare_data()
    datamodule.setup('fit')

    # Add information about the dataset to the experimental parameters
    config['num_train_samples'] = datamodule.num_train_samples
    config['num_val_samples'] = datamodule.num_val_samples
    config['num_train_batches'] = int(config['num_train_samples'] / config['batch_size'])

    # Initialize autoencoder model
    autoencoder = RegionCAE(config)

    # Initialize loggers to monitor training and validation
    logger = pl.loggers.TensorBoardLogger(
        config['log_directory'],
        name=config['experiment_name']
    )

    # Initialize the Trainer object
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=50,
        # auto_lr_find=(True if config['learning_rate'] is None else False),
        callbacks=[
            pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=10)
        ]
    )

    # trainer.tune(autoencoder)
    lr_finder = trainer.tuner.lr_find(autoencoder)
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Do some automatic parameter setting
    autoencoder.hparams.learning_rate = lr_finder.suggestion()
    print(autoencoder.hparams.learning_rate)

    # Train the model
    trainer.fit(autoencoder)


if __name__ == '__main__':
    main()
