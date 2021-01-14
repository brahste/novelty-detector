import sys
import yaml
import pytorch_lightning as pl

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
    config['Data-Format']['num_train_samples'] = datamodule.num_train_samples
    config['Data-Format']['num_val_samples'] = datamodule.num_val_samples

    # Initialize autoencoder model
    autoencoder = RegionCAE(config)

    # Initialize loggers to monitor training and validation
    logger = pl.loggers.TensorBoardLogger(
        config['Experiment-Format']['log_directory'],
        name=config['Experiment-Format']['experiment_name']
    )

    # Initialize the Trainer object
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=3
    )

    # Train the model
    trainer.fit(autoencoder, datamodule)


if __name__ == '__main__':
    main()
