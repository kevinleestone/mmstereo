# Copyright 2021 Toyota Research Institute.  All rights reserved.

import os

import cv2
from torch.utils.data import Dataset

from data.loader_helper import LoaderHelper
from data.sample import Sample, SampleElement, SampleMetadata, ElementType, ElementKeys

# Expected paths of sample elements.
LEFT_DIR = "left"
RIGHT_DIR = "right"
LEFT_DISPARITY_DIR = "left_disparity"
RIGHT_DISPARITY_DIR = "right_disparity"


class StereoDataset(Dataset):

    def __init__(self, idx, dataset_path, sim, transform=None):
        super().__init__()
        cv2.setNumThreads(0)

        self.id = idx
        self.dataset_path = dataset_path
        self.sim = sim

        self.images = sorted(os.listdir(os.path.join(dataset_path, LEFT_DIR)))
        self.npz = [os.path.splitext(filename)[0] + ".npz" for filename in self.images]
        self.png = [os.path.splitext(filename)[0] + ".png" for filename in self.images]

        self.transform = transform
        self.loader_helper = LoaderHelper()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = {}

        left_image = self.loader_helper.load_image(os.path.join(self.dataset_path, LEFT_DIR, self.images[idx]))
        right_image = self.loader_helper.load_image(os.path.join(self.dataset_path, RIGHT_DIR, self.images[idx]))
        height, width, _ = left_image.shape

        left_disparity = self.loader_helper.load_npz(os.path.join(self.dataset_path, LEFT_DISPARITY_DIR, self.npz[idx]),
                                                     fallback_none=True)
        right_disparity = self.loader_helper.load_npz(
                os.path.join(self.dataset_path, RIGHT_DISPARITY_DIR, self.npz[idx]), fallback_none=True)

        # Crop image to be divisible by 32 pixels to avoid having to pad inputs to network.
        height -= height % 32
        width -= width % 32
        left_image = left_image[:height, :width, :]
        data[ElementKeys.LEFT_RGB] = SampleElement(left_image, ElementType.COLOR_IMAGE)
        right_image = right_image[:height, :width, :]
        data[ElementKeys.RIGHT_RGB] = SampleElement(right_image, ElementType.COLOR_IMAGE)
        if left_disparity is not None:
            left_disparity = left_disparity[:height, :width]
            data[ElementKeys.LEFT_DISPARITY] = SampleElement(left_disparity, ElementType.DISPARITY_IMAGE)
        if right_disparity is not None:
            right_disparity = right_disparity[:height, :width]
            data[ElementKeys.RIGHT_DISPARITY] = SampleElement(right_disparity, ElementType.DISPARITY_IMAGE)

        sample = Sample(SampleMetadata(self.id, self.sim), data)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
