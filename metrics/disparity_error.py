# Copyright 2021 Toyota Research Institute.  All rights reserved.

from torchmetrics import Metric
import torch


class DisparityError(Metric):
    """Compute mean absolute error of output disparity compared to ground truth"""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(compute_on_step=False, dist_sync_on_step=dist_sync_on_step)

        self.add_state("error", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, disparity, disparity_gt, disparity_valid_mask):
        self.error += torch.sum(torch.abs(disparity - disparity_gt)[disparity_valid_mask])
        self.total += torch.sum(disparity_valid_mask)

    def compute(self):
        if self.total > 0:
            return self.error / self.total
        else:
            return torch.tensor(0)
