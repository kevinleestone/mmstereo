# Copyright 2021 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torch.nn.functional as F

from args import TrainingConfig
from data.sample import ElementKeys


def split_outputs(outputs):
    """Split a batch of merged left and right tensors back into individual left and right tensors"""
    batch_size, _, _, _ = outputs.shape
    assert batch_size % 2 == 0
    batch_size = batch_size // 2
    left_outputs = outputs[:batch_size]
    right_outputs = outputs[batch_size:]
    return left_outputs, right_outputs


def downsample_disparity(disparity, factor):
    """Downsample disparity using a min-pool operation

    Input can be either a Numpy array or Torch tensor.
    """
    with torch.no_grad():
        # Convert input to tensor at the appropriate number of dimensions if needed.
        is_numpy = type(disparity) == np.ndarray
        if is_numpy:
            disparity = torch.from_numpy(disparity)
        new_dims = 4 - len(disparity.shape)
        for i in range(new_dims):
            disparity = disparity.unsqueeze(0)

        disparity = F.max_pool2d(disparity, kernel_size=factor, stride=factor) / factor

        # Convert output disparity back into same format and number of dimensions as input.
        for i in range(new_dims):
            disparity = disparity.squeeze(0)
        if is_numpy:
            disparity = disparity.numpy()
        return disparity


def get_smoothness_image(hparams: TrainingConfig, batch_data, output):
    is_right = output.get("right", False)
    if is_right:
        return batch_data.get(
            ElementKeys.RIGHT_RGB_UNCORRUPTED, batch_data[ElementKeys.RIGHT_RGB]
        )
    else:
        return batch_data.get(
            ElementKeys.LEFT_RGB_UNCORRUPTED, batch_data[ElementKeys.LEFT_RGB]
        )


def get_max_disparity(hparams: TrainingConfig, batch_data, output):
    scale = output.get("scale", 1)
    return output.get("max_disparity", hparams.model.num_disparities // scale - 1)


def get_disparity_gt(hparams: TrainingConfig, batch_data, output):
    is_right = output.get("right", False)
    if is_right:
        disparity_gt = batch_data.get(ElementKeys.RIGHT_DISPARITY, None)
    else:
        disparity_gt = batch_data.get(ElementKeys.LEFT_DISPARITY, None)

    if disparity_gt is not None:
        # Scale ground truth disparity based on output scale.
        scale = output.get("scale", 1)
        disparity_gt = downsample_disparity(disparity_gt, scale)
    return disparity_gt


def get_disparity_valid_mask(
    hparams: TrainingConfig, batch_data, output, *, low_check_only=False
):
    is_right = output.get("right", False)
    disparity_gt = get_disparity_gt(hparams, batch_data, output)

    max_disparity = get_max_disparity(hparams, batch_data, output)
    ignore_edge = output.get("ignore_edge", False)

    low_check = disparity_gt > 1e-3
    if low_check_only:
        result = low_check
    else:
        result = torch.logical_and(low_check, disparity_gt < max_disparity)
    if ignore_edge:
        width = disparity_gt.shape[-1]
        edge_mask = (
            torch.arange(width, dtype=disparity_gt.dtype, device=disparity_gt.device)
            - 1
        )
        if is_right:
            edge_mask = torch.flip(edge_mask, (0,))
        edge_mask = edge_mask.expand_as(disparity_gt)
        valid_edge = disparity_gt < edge_mask
        result = torch.logical_and(result, valid_edge)
    return result
