# Copyright 2021 Toyota Research Institute.  All rights reserved.

import torch

from args import TrainingConfig
from metrics.disparity_correct import DisparityCorrect
from metrics.disparity_error import DisparityError
import utils


class Metrics(object):
    """Container to aid management of stereo network metrics"""

    def __init__(self, hparams: TrainingConfig):
        self.hparams = hparams
        self.metrics = {}

    def update_metrics(self, batch, outputs, trainer, key):
        batch_metadata, batch_data = batch

        for output_name, output in outputs.items():
            disparity = output.get("disparity", None)
            disparity_gt = utils.get_disparity_gt(self.hparams, batch_data, output)
            if disparity is not None and disparity_gt is not None and torch.max(disparity_gt) > 0.0:
                disparity_valid_mask = utils.get_disparity_valid_mask(self.hparams, batch_data, output)

                self.disparity_metrics(trainer, key, output_name, disparity, disparity_gt, disparity_valid_mask)

    def disparity_metrics(self, trainer, key, output_name, disparity, disparity_gt, disparity_valid_mask):
        self.error_metric(trainer, "{}_acc_{}/error".format(key, output_name), disparity, disparity_gt,
                          disparity_valid_mask)
        self.correct_metric(trainer, "{}_acc_{}/correct_0.25".format(key, output_name), disparity, disparity_gt,
                            disparity_valid_mask, 0.25)
        self.correct_metric(trainer, "{}_acc_{}/correct_0.5".format(key, output_name), disparity, disparity_gt,
                            disparity_valid_mask, 0.5)
        self.correct_metric(trainer, "{}_acc_{}/correct_1.0".format(key, output_name), disparity, disparity_gt,
                            disparity_valid_mask, 1.0)

    def error_metric(self, trainer, key, disparity, disparity_gt, disparity_valid_mask):
        if key not in self.metrics:
            self.metrics[key] = DisparityError().to(disparity.device)

        metric = self.metrics[key]
        metric.update(disparity, disparity_gt, disparity_valid_mask)
        trainer.log(key, metric, metric_attribute=trainer.metrics)

    def correct_metric(self, trainer, key, disparity, disparity_gt, disparity_valid_mask, threshold):
        if key not in self.metrics:
            self.metrics[key] = DisparityCorrect(threshold).to(disparity.device)

        metric = self.metrics[key]
        metric.update(disparity, disparity_gt, disparity_valid_mask)
        trainer.log(key, metric, metric_attribute=trainer.metrics)

    def reset(self):
        for _, metric in self.metrics.items():
            metric.reset()
