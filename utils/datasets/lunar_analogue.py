# Datasets used in Novelty detection experiments
# Author: Braden Stefanuk
# Created: Dec 17, 2020

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from pathlib import Path
from torchvision import transforms
from skimage import io, transform
from utils import tools
from typing import Optional, Generic

# !Temporary constants
ROOT_DATA_PATH = '/home/brahste/Datasets/LunarAnalogue/images-screened'


class LunarAnalogueDataset(torch.utils.data.Dataset):
    '''Creating a map style dataset of the lunar analogue terrain'''

    def __init__(
            self,
            data_config: dict,
            train: bool = True,
            transforms: Optional[transforms.Compose] = None
    ):
        super(LunarAnalogueDataset, self).__init__()

        # We handle the training and testing data with various glob
        # patterns, this helps us be able to adapt and implement
        # alternative labelling scheme
        if train:
            self._glob_pattern = data_config['glob_pattern_train']
        else:
            self._glob_pattern = data_config['glob_pattern_test']

        self._root_data_path = Path(data_config['root_data_path'])
        self._list_of_image_paths = list(self._root_data_path.glob(self._glob_pattern))
        self._transforms = transforms

    def __len__(self):
        '''Returns the total number of images in the dataset'''
        return len(self._list_of_image_paths)

    def __getitem__(self, idx: int):

        image = io.imread(self._list_of_image_paths[idx])

        if self._transforms:
            image = self._transforms(image)

        return image


class LunarAnalogueDataModule(pl.core.datamodule.LightningDataModule):
    def __init__(self, config: dict):
        super(LunarAnalogueDataModule, self).__init__()

        # Unpack just the 'Data-Format' section of the configuration
        self._config = config
        self._root_data_path = self._config['root_data_path']
        self._batch_size = self._config['batch_size']
        self._train_fraction = self._config['train_fraction']
        self._val_fraction = 1 - self._config['train_fraction']

        self._transforms = transforms.Compose([
            tools.PreprocessingPipeline(),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        '''
        Prepare the data by cascading processing operations to be conducted
        during import
        '''
        return

    def setup(self, stage: Optional[str] = None):
        '''
        Prepare the data by cascading processing operations to be conducted
        during import
        '''
        if stage == 'fit' or stage is None:
            # Setup training and validation data for use in dataloaders
            dataset_trainval = LunarAnalogueDataset(
                self._config,
                train=True,
                transforms=self._transforms
            )
            # Calculate and save values for use in the training program
            self.num_train_samples = int(np.floor(len(dataset_trainval) * self._train_fraction))
            self.num_val_samples = int(np.floor(len(dataset_trainval) * self._val_fraction))

            self._dataset_train, self._dataset_val = torch.utils.data.random_split(
                dataset_trainval,
                [self.num_train_samples, self.num_val_samples]
            )

        if stage == 'test' or stage is None:
            # Setup testing data as well
            self._dataset_test = LunarAnalogueDataset(
                self._config,
                train=False,
                transforms=self._transforms
            )
        return

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._dataset_train,
            batch_size=self._batch_size,
            drop_last=True,
            num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._dataset_val,
            batch_size=self._batch_size,
            drop_last=True,
            num_workers=4
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._dataset_test,
            batch_size=self._batch_size,
            drop_last=True,
            num_workers=4
        )


if __name__ == '__main__':
    # dataset = LunarAnalogueDataset('/home/brahste/Datasets/LunarAnalogue/images-screened')
    # print(len(dataset))
    # print(type(dataset[44]))
    # print(dataset[44].shape)
    # # plt.imshow(dataset[550]); plt.show();

    dataloader = LunarAnalogueDataModule().train_dataloader()

    print(dataloader)
    print(type(dataloader))
    print(dir(dataloader))

    train1 = next(iter(dataloader))
    print(train1.shape)

    # for i, batch in enumerate(dataloader):
    #     print(batch.shape)
    #     # plt.imshow(image); plt.show()
    #     if i > 10 : break
