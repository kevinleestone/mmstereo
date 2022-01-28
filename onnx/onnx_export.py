# Copyright 2021 Toyota Research Institute.  All rights reserved.

import copy

import torch
import torch.nn as nn

from args import TrainingConfig
from layers.cost_volume import CostVolume
from layers.matchability import Matchability
from layers.soft_argmin import SoftArgmin
from onnx.onnx_plugins import ExportableCostVolume, ExportableMatchability, ExportableSoftArgmin, ExportableFlatten


class ExportableStereo(nn.Module):

    def __init__(self, hparams: TrainingConfig, model):
        super().__init__()
        self.model = model

        # This needs the weird shape due to issues with TensorRT 7.0.
        self.normalization_factor = torch.tensor([[[[1.0 / 255.0]]]], dtype=torch.float32)

    def forward(self, left_image, right_image):
        # Apply the normalization factor needed for inference in FLT inference modules.
        normalized_left = left_image * self.normalization_factor
        normalized_right = right_image * self.normalization_factor

        output, _ = self.model(normalized_left, normalized_right)
        disparity = output["disparity"]
        disparity_small = output["disparity_small"]
        matchability = output["matchability"]
        return disparity, disparity_small, torch.exp(matchability)


def fix_module(module):
    """Recursively replace modules with ONNX-compatible ones"""
    for child_module_name, child_module in module.named_children():
        if isinstance(child_module, CostVolume):
            num_disparities = child_module.num_disparities
            is_right = child_module.is_right
            module._modules[child_module_name] = ExportableCostVolume(num_disparities, is_right)
        elif isinstance(child_module, Matchability):
            module._modules[child_module_name] = ExportableMatchability()
        elif isinstance(child_module, SoftArgmin):
            module._modules[child_module_name] = ExportableSoftArgmin()
        elif isinstance(child_module, nn.Flatten):
            start_dim = int(child_module.start_dim)
            end_dim = int(child_module.end_dim)
            module._modules[child_module_name] = ExportableFlatten(start_dim, end_dim)
        elif len(list(child_module.children())) > 0:
            fix_module(child_module)


def export_stereo_model(hparams: TrainingConfig, model, filename, height=544, width=960):
    """Create exportable model and write to given filename"""
    model = copy.deepcopy(model).cpu()
    fix_module(model)
    model = ExportableStereo(hparams, model)

    dummy_input = (torch.zeros((1, 3, height, width), dtype=torch.float32),
                   torch.zeros((1, 3, height, width), dtype=torch.float32))

    input_names = ["left_input", "right_input"]
    output_names = ["disparity", "disparity_small", "matchability"]

    torch.onnx.export(model, dummy_input, filename, verbose=False, input_names=input_names, output_names=output_names,
                      do_constant_folding=True, enable_onnx_checker=True, opset_version=11)
