# Copyright 2021 Toyota Research Institute.  All rights reserved.

from args import TrainingConfig
from losses.disparity_loss import DisparityLoss
from losses.nsce_loss import NsceLoss
from losses.smoothness_loss import SmoothnessLoss


class Losses(object):

    def __init__(self, hparams: TrainingConfig):
        super().__init__()

        self.hparams = hparams

        self.disparity_loss = DisparityLoss(hparams)
        self.nsce_loss = NsceLoss(hparams)
        self.smoothness_loss = SmoothnessLoss(hparams)

    def accumulate_loss(self, loss, new_loss, should_log, new_should_log):
        if new_loss is None:
            return loss, should_log
        if loss is None:
            loss = new_loss
        else:
            loss += new_loss
        should_log = should_log or new_should_log
        return loss, should_log

    def __call__(self, batch, outputs, trainer, key):
        all_loss = None
        all_should_log = False

        for output_name, output in outputs.items():
            single_output_loss = None
            single_should_log = False

            if self.hparams.loss.disparity_mult > 0.0:
                disparity_loss, should_log = self.disparity_loss(batch, output)
                if disparity_loss is not None:
                    disparity_loss = disparity_loss * self.hparams.loss.disparity_mult
                    if should_log:
                        trainer.log("{}_loss_disparity/{}".format(key, output_name), disparity_loss.item())
                    single_output_loss, single_should_log = self.accumulate_loss(single_output_loss, disparity_loss,
                                                                                 single_should_log, should_log)

            if self.hparams.loss.nsce_mult > 0.0:
                nsce_loss, should_log = self.nsce_loss(batch, output)
                if nsce_loss is not None:
                    nsce_loss = nsce_loss * self.hparams.loss.nsce_mult
                    if should_log:
                        trainer.log("{}_loss_nsce/{}".format(key, output_name), nsce_loss.item())
                    single_output_loss, single_should_log = self.accumulate_loss(single_output_loss, nsce_loss,
                                                                                 single_should_log, should_log)

            if self.hparams.loss.smoothness_mult > 0.0:
                smoothness_loss, should_log = self.smoothness_loss(batch, output)
                if smoothness_loss is not None:
                    smoothness_loss = smoothness_loss * self.hparams.loss.smoothness_mult
                    if should_log:
                        trainer.log("{}_loss_smoothness/{}".format(key, output_name), smoothness_loss.item())
                    single_output_loss, single_should_log = self.accumulate_loss(single_output_loss, smoothness_loss,
                                                                                 single_should_log, should_log)

            if single_should_log:
                trainer.log("{}_loss/{}".format(key, output_name), single_output_loss.item())

            all_loss, all_should_log = self.accumulate_loss(all_loss, single_output_loss, all_should_log,
                                                            single_should_log)

        if all_should_log:
            trainer.log("{}_loss".format(key), all_loss.item())
        return all_loss
