# Copyright 2021 Toyota Research Institute.  All rights reserved.

import kornia
import torch.nn as nn
import torch.nn.functional as F

from args import TrainingConfig
from losses.loss_utils import null_loss, valid_loss
import utils


class SmoothnessLoss(nn.Module):

    def __init__(self, hparams: TrainingConfig):
        super().__init__()

        self.hparams =hparams
        self.loss = kornia.losses.InverseDepthSmoothnessLoss()

    def forward(self, batch, output):
        batch_metadata, batch_data = batch

        scale = output.get("scale", 1)
        if scale == 1:
            return null_loss()

        disparity = output.get("disparity")
        if disparity is None:
            return null_loss()

        max_disparity = utils.get_max_disparity(self.hparams, batch_data, output)

        image = utils.get_smoothness_image(self.hparams, batch_data, output)
        image = F.interpolate(image, size=disparity.shape[2:], mode='bilinear', align_corners=False)

        return valid_loss(self.loss(disparity / max_disparity, image))
