# Copyright 2021 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
from pytorch_lightning.utilities import rank_zero_only
import torch

from args import TrainingConfig
from data.sample import ElementKeys
from logger_utils import get_tensorboard
import utils


def make_bgr_image(image, idx=0):
    """Generate visualization as tensor for a BGR image stored as a tensor"""
    vis_image = torch.zeros_like(image[idx, :, :, :])
    vis_image[0, :, :] = image[idx, 2, :, :]
    vis_image[1, :, :] = image[idx, 1, :, :]
    vis_image[2, :, :] = image[idx, 0, :, :]
    return vis_image


def make_cv_disparity_image(disparity, max_disparity):
    """Generate a disparity visualization as a numpy array using OpenCV's jet color mapping"""
    vis_disparity = disparity / max_disparity
    vis_disparity[vis_disparity < 0.0] = 0.0
    vis_disparity[vis_disparity > 1.0] = 1.0
    vis_disparity = vis_disparity.cpu()
    np_img = (vis_disparity.numpy() * 255.0).astype(np.uint8)
    mapped = cv2.applyColorMap(np_img, cv2.COLORMAP_JET)
    mapped[vis_disparity < 1e-3, :] = 0
    mapped[vis_disparity > 1.0 - 1e-3, :] = 0
    return mapped


def make_cv_normalized_disparity_image(disparity, disparity_gt, max_disparity):
    """Generate a disparity visualization as a numpy array using OpenCV's jet color mapping"""
    try:
        min_disparity = torch.clamp(torch.min(disparity_gt[disparity_gt > 1e-3]) - 0.1, min=1e-3)
        max_disparity = torch.clamp(torch.max(disparity_gt[disparity_gt > 1e-3]) + 0.1, min=1e-3)
    except RuntimeError:
        min_disparity = 0.0
        max_disparity = max_disparity
    vis_disparity = (disparity - min_disparity) / (max_disparity - min_disparity)
    vis_disparity[vis_disparity < 0.0] = 0.0
    vis_disparity[vis_disparity > 1.0] = 1.0
    vis_disparity = vis_disparity.cpu()
    np_img = (vis_disparity.numpy() * 255.0).astype(np.uint8)
    mapped = cv2.applyColorMap(np_img, cv2.COLORMAP_JET)
    mapped[vis_disparity < 1e-3, :] = 0
    mapped[vis_disparity > 1.0 - 1e-3, :] = 0
    return mapped


def make_disparity_image(hparams: TrainingConfig, batch_data, output, idx=0):
    """Generate a disparity visualization as a tensor using OpenCV's jet color mapping"""
    disparity = output["disparity"][idx:idx + 1, :, :, :].clone().detach().to(torch.float32)
    max_disparity = utils.get_max_disparity(hparams, batch_data, output)
    mapped = make_cv_disparity_image(disparity[idx, 0, :, :], max_disparity)
    mapped = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    mapped = torch.from_numpy(mapped).permute(2, 0, 1)
    return mapped


def make_normalized_disparity_image(hparams: TrainingConfig, batch_data, output, idx=0):
    """Generate a disparity visualization as a tensor using OpenCV's jet color mapping"""
    disparity = output["disparity"][idx:idx + 1, :, :, :].clone().detach().to(torch.float32)
    max_disparity = utils.get_max_disparity(hparams, batch_data, output)
    disparity_gt = utils.get_disparity_gt(hparams, batch_data, output)
    if disparity_gt is not None:
        disparity_gt = disparity_gt[idx:idx + 1, :, :, :]
        valid_mask = utils.get_disparity_valid_mask(hparams, batch_data, output)[idx:idx + 1, :, :, :]
        disparity_gt[torch.logical_not(valid_mask)] = 0.0
    else:
        disparity_gt = disparity
    mapped = make_cv_normalized_disparity_image(disparity[idx, 0, :, :], disparity_gt[idx, 0, :, :], max_disparity)
    mapped = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    mapped = torch.from_numpy(mapped).permute(2, 0, 1)
    return mapped


def make_disparity_gt_image(hparams: TrainingConfig, batch_data, output, idx=0):
    """Generate a disparity visualization as a tensor using OpenCV's jet color mapping"""
    disparity_gt = utils.get_disparity_gt(hparams, batch_data, output)[idx:idx + 1, :, :, :]
    valid_mask = utils.get_disparity_valid_mask(hparams, batch_data, output)[idx:idx + 1, :, :, :]
    disparity_gt[torch.logical_not(valid_mask)] = 0.0
    max_disparity = utils.get_max_disparity(hparams, batch_data, output)
    mapped = make_cv_disparity_image(disparity_gt[idx, 0, :, :], max_disparity)
    mapped = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    mapped = torch.from_numpy(mapped).permute(2, 0, 1)
    return mapped


def make_normalized_disparity_gt_image(hparams: TrainingConfig, batch_data, output, idx=0):
    """Generate a disparity visualization as a tensor using OpenCV's jet color mapping"""
    disparity_gt = utils.get_disparity_gt(hparams, batch_data, output)[idx:idx + 1, :, :, :]
    max_disparity = utils.get_max_disparity(hparams, batch_data, output)
    valid_mask = utils.get_disparity_valid_mask(hparams, batch_data, output)[idx:idx + 1, :, :, :]
    disparity_gt[torch.logical_not(valid_mask)] = 0.0
    mapped = make_cv_normalized_disparity_image(disparity_gt[idx, 0, :, :], disparity_gt[idx, 0, :, :], max_disparity)
    mapped = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    mapped = torch.from_numpy(mapped).permute(2, 0, 1)
    return mapped


def make_cv_confidence_image(confidence):
    """Generate a disparity confidence visualization as a numpy array using OpenCV's jet color mapping"""
    vis_confidence = confidence.clone().detach().cpu()
    np_img = (vis_confidence.numpy() * 255.0).astype(np.uint8)
    mapped = cv2.applyColorMap(np_img, cv2.COLORMAP_JET)
    return mapped


def make_confidence_image(confidence, idx=0):
    """Generate a disparity confidence visualization as a tensor using OpenCV's jet color mapping"""
    mapped = make_cv_confidence_image(confidence[idx, 0, :, :])
    mapped = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    mapped = torch.from_numpy(mapped).permute(2, 0, 1)
    return mapped


def make_dummy_image():
    return torch.zeros(3, 1, 1)


def log_disparity(all_loggers, hparams: TrainingConfig, step, prefix, suffix, batch_data, output, output_name):
    tensorboard = get_tensorboard(all_loggers).experiment

    scale = output.get("scale", 1)
    disparity = output.get("disparity", None)
    if disparity is not None:
        disp_image = make_disparity_image(hparams, batch_data, output)
        tensorboard.add_image("{}{}/disparity_{}/output".format(prefix, suffix, output_name), disp_image, step)

        disparity_gt = utils.get_disparity_gt(hparams, batch_data, output)
        if disparity_gt is not None:
            disparity_gt_vis = make_disparity_gt_image(hparams, batch_data, output)
        else:
            disparity_gt_vis = make_dummy_image()
        tensorboard.add_image("{}{}/disparity_{}/gt".format(prefix, suffix, output_name), disparity_gt_vis, step)

        if scale == 1 and "refine" in output_name:
            tensorboard.add_image("{}{}/normalized_disparity_{}/output".format(prefix, suffix, output_name),
                                  make_normalized_disparity_image(hparams, batch_data, output), step)

            if disparity_gt is not None:
                normalized_disparity_gt_vis = make_normalized_disparity_gt_image(hparams, batch_data, output)
            else:
                normalized_disparity_gt_vis = make_dummy_image()
            tensorboard.add_image("{}{}/normalized_disparity_{}/gt".format(prefix, suffix, output_name),
                                  normalized_disparity_gt_vis, step)


@rank_zero_only
def log_images(all_loggers, hparams: TrainingConfig, step, prefix, batch, all_outputs):
    """Visualize image outputs in Tensorboard"""
    batch_metadata, batch_data = batch

    logger = get_tensorboard(all_loggers).experiment

    logger.add_image("{}/input_left".format(prefix), make_bgr_image(batch_data[ElementKeys.LEFT_RGB]), step)
    logger.add_image("{}/input_right".format(prefix), make_bgr_image(batch_data[ElementKeys.RIGHT_RGB]), step)

    for idx, (output_name, output) in enumerate(all_outputs.items()):
        right = output.get("right", False)
        if right and not hparams.vis_right:
            continue
        suffix = "_right" if right else ""

        scale_factor = output["scale"]
        if "refine" in output_name and scale_factor != 1:
            continue

        log_disparity(all_loggers, hparams, step, prefix, suffix, batch_data, output, output_name)

        # Matchability is an optional output, so check for None result.
        matchability = output.get("matchability", None)
        if matchability is not None:
            logger.add_image('{}{}/matchability_{}'.format(prefix, suffix, output_name),
                             make_confidence_image(torch.exp(matchability)), step)
