# Copyright 2021 Toyota Research Institute.  All rights reserved.

import os

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


class OnnxExport(Callback):
    """Callback for automatically exporting ONNX to the checkpoint directory at the end of validation"""

    def __init__(self, output_dir, note):
        self.output_dir = output_dir
        self.note = note

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        path = os.path.join(self.output_dir, "model.onnx")
        pl_module.export_onnx(path)


class TorchscriptExport(Callback):
    """Callback for automatically exporting Torchscript to the checkpoint directory at the end of validation"""

    def __init__(self, output_dir, note):
        self.output_dir = output_dir
        self.note = note

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        path = os.path.join(self.output_dir, "model.pt")
        pl_module.export_torchscript(path)
