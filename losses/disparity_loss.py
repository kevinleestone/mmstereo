# Copyright 2021 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from args import TrainingConfig
from losses.loss_utils import null_loss, dummy_loss, valid_loss
import utils


class DisparityLoss(nn.Module):
    """Smooth L1-loss for disparity with check for valid ground truth"""

    def __init__(self, hparams: TrainingConfig):
        super().__init__()

        self.hparams = hparams
        self.loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, batch, output):
        batch_metadata, batch_data = batch

        disparity = output.get("disparity", None)
        if disparity is None:
            return null_loss()
        disparity_gt = utils.get_disparity_gt(self.hparams, batch_data, output)
        if disparity_gt is None:
            return dummy_loss(disparity)

        valid_mask = utils.get_disparity_valid_mask(self.hparams, batch_data, output)

        batch_size, _, height, width = disparity.shape
        loss = torch.tensor(0, dtype=disparity.dtype, device=disparity.device)

        per_frame_valid_threshold = height * width * 0.03

        # Not all batch elements may have ground truth for disparity, so we compute the loss for each batch element
        # individually.
        valid_count = 0
        for batch_idx in range(batch_size):
            if torch.sum(valid_mask[batch_idx, :, :, :]) < per_frame_valid_threshold:
                # print("wtf", torch.sum(valid_mask[batch_idx, :, :, :]), per_frame_valid_threshold)
                continue

            single_loss = self.loss(disparity[batch_idx, :, :, :], disparity_gt[batch_idx, :, :, :])[
                valid_mask[batch_idx, :, :, :]]
            valid_count += 1

            if self.hparams.loss.disparity_stdmean_scaled:
                # Scale loss by standard deviation and mean of ground truth to reduce influence of very high
                # disparities.
                gt_std, gt_mean = torch.std_mean(disparity_gt[batch_idx, :, :, :][valid_mask[batch_idx, :, :, :]])
                loss += torch.mean(single_loss) / (gt_mean + 2.0 * gt_std)
            else:
                # Scale loss by scale factor due to difference of expected magnitude of disparity at different scales.
                scale = output.get("scale", None)
                loss += torch.mean(single_loss) * scale

        # Avoid potential divide by 0.
        if valid_count > 0:
            return valid_loss(loss / batch_size)
        else:
            return dummy_loss(disparity)
