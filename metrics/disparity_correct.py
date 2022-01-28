# Copyright 2021 Toyota Research Institute.  All rights reserved.

from torchmetrics import Metric
import torch


class DisparityCorrect(Metric):
    """Compute fraction of disparity outputs matches compared to ground truth within given threshold"""

    def __init__(self, threshold, dist_sync_on_step=False):
        super().__init__(compute_on_step=False, dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold

        self.add_state("correct", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, disparity, disparity_gt, disparity_valid_mask):
        diff = torch.abs(disparity - disparity_gt)[disparity_valid_mask]
        self.correct += torch.sum(diff <= self.threshold)
        self.total += torch.sum(disparity_valid_mask)

    def compute(self):
        if self.total > 0:
            return self.correct.to(torch.float64) / self.total
        else:
            return torch.tensor(0)
