# Copyright 2021 Toyota Research Institute.  All rights reserved.

import random

import pytorch_lightning as pl
import torch
import torch.distributed
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from args import DataConfig, StageConfig, TransformConfig
from data.sample import collate
from data.stereo_dataset import StereoDataset
from data.stereo_transforms import RandomCrop, RandomHorizontalFlip, RandomColorJitter, NormalizeImage, ToTensor, \
    Compose, KeepUncorrupted


class StereoDataModule(pl.LightningDataModule):

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config

        self.train = None
        self.val = None

    def setup(self, stage=None):
        assert stage is None or stage == "fit"

        self.train = self.get_dataset("train", self.config.train)
        self.val = self.get_dataset("val", self.config.val)

    def train_dataloader(self):
        return self.get_loader(self.config.train, self.train)

    def val_dataloader(self):
        return self.get_loader(self.config.val, self.val)

    def get_dataset(self, prefix, config: StageConfig):
        transform = self.get_transform(config.transform)

        datasets = []
        for idx, dataset_config in enumerate(config.datasets):
            dataset = StereoDataset(idx, dataset_config.path, dataset_config.sim, transform)
            total_size = len(dataset)

            if not dataset_config.no_split:
                # Split dataset into train and validation sets (90% train, 10% val).
                train_size = int(round(total_size * 0.9))
                val_size = total_size - train_size
                train, val = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
                if prefix == "train":
                    dataset = train
                else:
                    dataset = val

            if dataset_config.fraction != 1.0:
                partial_size = int(round(len(dataset) * dataset_config.fraction))
                dataset = Subset(dataset, range(partial_size))

            if dataset_config.repeats != 1:
                dataset = ConcatDataset([dataset] * dataset_config.repeats)

            datasets.append(dataset)

        return ConcatDataset(datasets)

    def get_loader(self, stage_config: StageConfig, dataset):
        generator = torch.Generator()
        generator.manual_seed(12345)
        return DataLoader(dataset, batch_size=stage_config.batch_size, shuffle=stage_config.shuffle,
                          num_workers=stage_config.num_workers, collate_fn=collate, pin_memory=False, drop_last=True,
                          generator=generator)

    def get_transform(self, config: TransformConfig):
        transform_list = []

        seed = 0

        if config.random_crop is not None:
            generator, seed = self._get_generator(seed)
            transform_list.append(RandomCrop(config.random_crop, generator))

        if config.random_horizontal_flip:
            generator, seed = self._get_generator(seed)
            transform_list.append(RandomHorizontalFlip(generator))

        if config.keep_uncorrupted:
            transform_list.append(KeepUncorrupted())

        if config.random_color_jitter:
            generator, seed = self._get_generator(seed)
            transform_list.append(RandomColorJitter(generator))

        # All samples require normalization and conversion to tensor as the last operation.
        mean = [0.0, 0.0, 0.0]
        scale = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        transform_list.append(NormalizeImage({'mean': mean, 'scale': scale}))
        transform_list.append(ToTensor())

        return Compose(transform_list)

    def _get_generator(self, seed):
        return random.Random(seed), seed + 1
