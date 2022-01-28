# Copyright 2021 Toyota Research Institute.  All rights reserved.

import random

import pytorch_lightning as pl
import torch

from args import TrainingConfig
import model_loader
from data.sample import ElementKeys, SampleMetadata
from data.stereo_batch_transforms import CameraEffect
from losses.losses import Losses
from metrics.metrics import Metrics
from onnx.onnx_export import export_stereo_model
from optim.poly_lr import lambda_poly_lr
import visualization


class StereoModel(pl.LightningModule):

    def __init__(self, hparams: TrainingConfig):
        super().__init__()
        self.config = hparams
        self.save_hyperparameters(self.config)
        self.validation_visualized = []

        self.model = model_loader.load_model(hparams.model)
        if hparams.sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Losses.
        self.losses = Losses(hparams)

        # Metrics.
        self.metrics = Metrics(hparams)

        # self.image_log_mod = 10 if self.hparams.overfit else 50
        self.image_log_mod = 50

        # Batch augmentations.
        if hparams.random_camera_effect:
            self.batch_transform = CameraEffect(random.Random(0))
        else:
            self.batch_transform = None

    def forward(self, batch):
        # Extract inputs from batch and run forward pass.
        batch_metadata, batch_data = batch
        left_image = batch_data[ElementKeys.LEFT_RGB]
        right_image = batch_data[ElementKeys.RIGHT_RGB]
        output, all_outputs = self.model(left_image, right_image)
        return output, all_outputs

    def training_step(self, batch, batch_idx):
        if isinstance(batch[0], tuple):
            batch = SampleMetadata(*batch[0]), batch[1]
        # Apply batch transformations to the training batch if necessary.
        if self.batch_transform is not None:
            with torch.no_grad():
                batch = self.batch_transform(batch)

        output, all_outputs = self(batch)

        # Generate losses with network output.
        loss = self.losses(batch, all_outputs, self, "train")

        # Visualize every n batches during training.
        with torch.no_grad():
            if batch_idx % self.image_log_mod == 0:
                visualization.log_images(self.logger, self.config, self.global_step, "train", batch, all_outputs)

        return loss

    def on_validation_epoch_start(self):
        # Reset the list of visualized datasets.
        self.validation_visualized = []
        # PyTorch lightning doesn't seem to reset the metrics after the validation sanity check,
        # so we reset it manually here.
        self.metrics.reset()

    def validation_step(self, batch, batch_idx):
        if isinstance(batch[0], tuple):
            batch = SampleMetadata(*batch[0]), batch[1]
        batch_metadata, batch_data = batch

        # Figure out what dataset this specific validation sample comes from.
        assert len(batch_metadata.dataset_id) == 1
        dataset_id = batch_metadata.dataset_id[0].item()
        key = "val" + str(dataset_id)

        with torch.cuda.amp.autocast(enabled=False):
            output, all_outputs = self(batch)

        loss = self.losses(batch, all_outputs, self, key)
        if loss is not None:
            if dataset_id == 0:
                self.log("val_loss", loss.item())

            self.metrics.update_metrics(batch, all_outputs, self, key)
        else:
            loss = torch.tensor(0)

        # Visualize the validation result once per epoch.
        if dataset_id not in self.validation_visualized and self.global_step > 0:
            visualization.log_images(self.logger, self.config, self.current_epoch, key, batch, all_outputs)
            self.validation_visualized.append(dataset_id)

        return loss

    def configure_optimizers(self):
        # Select optimizer based on hparams.
        if self.config.optimizer.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.config.optimizer.learning_rate,
                                        momentum=self.config.optimizer.momentum,
                                        weight_decay=self.config.optimizer.weight_decay)
        elif self.config.optimizer.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.config.optimizer.learning_rate,
                                         weight_decay=self.config.optimizer.weight_decay)
        else:
            raise RuntimeError()

        # Select learning rate policy (if any) based on hparams.
        if self.config.optimizer.lr_policy == "poly":
            lr_lambda = lambda_poly_lr(self.config.epochs, self.config.optimizer.poly_exp)
            schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)]
        else:
            schedulers = []

        return [optimizer], schedulers

    def export_onnx(self, filename, height=2048, width=2560):
        """Export model to ONNX that can be imported into TensorRT"""
        with torch.cuda.amp.autocast(enabled=False):
            exportable_model = model_loader.get_cpu_model_copy(self.config.model, self.model)
            export_stereo_model(self.config, exportable_model, filename, height, width)

    def export_torchscript(self, filename):
        """Export Torchscript module that can be used for easy stereo inference in Python"""
        with torch.cuda.amp.autocast(enabled=False):
            exportable_model = model_loader.get_cpu_model_copy(self.config.model, self.model)
            script = torch.jit.script(exportable_model)
            torch.jit.save(script, filename)
