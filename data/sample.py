# Copyright 2021 Toyota Research Institute.  All rights reserved.

from collections import namedtuple
from enum import Enum

import cv2
import torch
from torch.utils.data.dataloader import default_collate


class ElementKeys(Enum):
    """Keys for looking up specific data in a samnple"""
    VALID = 0
    LEFT_RGB = 1
    RGB = LEFT_RGB
    RIGHT_RGB = 2
    LEFT_DISPARITY = 3
    DISPARITY = LEFT_DISPARITY
    RIGHT_DISPARITY = 4
    LEFT_RGB_UNCORRUPTED = 7
    RGB_UNCORRUPTED = LEFT_RGB_UNCORRUPTED
    RIGHT_RGB_UNCORRUPTED = 8


class ElementType(Enum):
    """Types of data that can be stored in a data sample"""
    COLOR_IMAGE = 0
    DISPARITY_IMAGE = 1


class SampleElement(namedtuple("SampleElement", ["data", "type"])):
    def interpolation_type(self):
        """Get what interpolation mode should be used when resizing this element"""
        if self.type == ElementType.COLOR_IMAGE:
            return cv2.INTER_LINEAR
        elif self.type == ElementType.DISPARITY_IMAGE:
            return cv2.INTER_NEAREST
        else:
            raise RuntimeError()

    def should_normalize(self):
        """Check if this element should be normalized during data loading"""
        return self.type == ElementType.COLOR_IMAGE

    def should_transpose(self):
        """Check if this element should be transposed to HWC to CHW layout during data loading"""
        return self.type == ElementType.COLOR_IMAGE

    def is_disparity(self):
        """Check if this element is a disparity image"""
        return self.type == ElementType.DISPARITY_IMAGE

    def is_image(self):
        """Check if this element is an image (2D array with 1 or 3 channels)"""
        return True

    def tensor_type(self):
        """Get what type of tensor this element should be when being converted from Numpy arrays"""
        return torch.float32


SampleMetadata = namedtuple("SampleMetadata", ["dataset_id", "simulated"])


class Sample(namedtuple("Sample", ["metadata", "elements"])):

    def get_data(self, key):
        if key in self.elements:
            return self.elements[key].data
        else:
            return None

    def set_data(self, key, new_data):
        element = self.elements[key]
        self.elements[key] = SampleElement(new_data, element.type)

    def has_disparity(self):
        return ElementKeys.LEFT_DISPARITY in self.elements or ElementKeys.RIGHT_DISPARITY in self.elements

    def is_flippable(self):
        return not self.is_stereo or (not self.has_disparity() or (
                    ElementKeys.LEFT_DISPARITY in self.elements and ElementKeys.RIGHT_DISPARITY in self.elements))

    def is_stereo(self):
        return ElementKeys.LEFT_RGB in self.elements and ElementKeys.RIGHT_RGB in self.elements

    def swap_left_right(self):
        assert self.is_flippable()

        def swap(key_left, key_right):
            if key_left in self.elements and key_right in self.elements:
                self.elements[key_left], self.elements[key_right] = self.elements[key_right], self.elements[key_left]

        swap(ElementKeys.LEFT_RGB, ElementKeys.RIGHT_RGB)
        swap(ElementKeys.LEFT_DISPARITY, ElementKeys.RIGHT_DISPARITY)


def collate(samples):
    first_elements = {}
    for sample in samples:
        for key, element in sample.elements.items():
            if key not in first_elements:
                first_elements[key] = element.data
            else:
                assert element.data.shape == first_elements[key].shape

    result_metadata = []
    result = []
    for sample in samples:
        result_metadata.append(sample.metadata)

        result_sample = {}
        result_valid = {}
        for key, first_element in first_elements.items():
            data = sample.get_data(key)
            if data is not None:
                result_sample[key] = data
                result_valid[key] = True
            else:
                if first_element.dtype == torch.int64:
                    result_sample[key] = torch.full_like(first_element, 255)
                else:
                    result_sample[key] = torch.zeros_like(first_element)
                result_valid[key] = False
        result_sample[ElementKeys.VALID] = result_valid
        result.append(result_sample)

    return default_collate(result_metadata), default_collate(result)
