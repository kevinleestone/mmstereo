# Copyright 2021 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from args import TrainingConfig
from losses.loss_utils import null_loss, dummy_loss, valid_loss
import utils

LAMBDA = 0.3


class NsceLoss(nn.Module):
    """Noise Sampling Cross Entropy Loss

    From https://arxiv.org/abs/2005.08806.
    This loss encourages a strong peak at the true disparity value to give a more unimodal output.
    """

    def __init__(self, hparams: TrainingConfig):
        super().__init__()

        self.hparams = hparams

    def forward(self, batch, output):
        batch_metadata, batch_data = batch

        cost_volume = output.get("cost_volume", None)
        if cost_volume is None:
            return null_loss()
        disparity_gt = utils.get_disparity_gt(self.hparams, batch_data, output)
        if disparity_gt is None:
            return dummy_loss(cost_volume)

        max_disparity = utils.get_max_disparity(self.hparams, batch_data, output)
        valid_mask = utils.get_disparity_valid_mask(self.hparams, batch_data, output)

        # Equation 12 from paper.
        batch_size, channels, height, width = cost_volume.shape
        a = torch.arange(1, max_disparity + 1, dtype=cost_volume.dtype, device=cost_volume.device).reshape(1, -1, 1, 1)
        a = torch.abs(a.expand(-1, -1, height, width) - disparity_gt) / LAMBDA
        a = torch.softmax(-a, dim=1)
        # Using log_softmax is more stable than calling log on softmax.
        b = torch.log_softmax(-cost_volume[:, 1:, :, :], dim=1)
        # Sum over disparity dimension.
        loss = -torch.sum(a * b, dim=1, keepdim=True)

        # Only consider valid disparities when computing loss.
        loss = torch.mean(loss[valid_mask])
        return valid_loss(loss)
