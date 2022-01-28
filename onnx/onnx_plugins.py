# Copyright 2021 Toyota Research Institute.  All rights reserved.

"""Modules that replace layers that aren't able to be exported from PyTorch to TensorRT"""

import struct

import torch
import torch.nn as nn

from layers.cost_volume import cost_volume
from layers.matchability import matchability
from layers.soft_argmin import soft_argmin


@torch.autograd.function.traceable
class ExportableCostVolumeFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, left, right, num_disparities, is_right):
        assert not is_right
        serialized_data = struct.pack("<iiiii", num_disparities, 0, 0, 0, 0)
        return g.op("trt::TRT_PluginV2", left, right, version_s="0.0.1", namespace_s="", data_s=serialized_data,
                    name_s="CrossCorrelationCostVolume")

    @staticmethod
    def forward(ctx, left, right, num_disparities, is_right):
        return cost_volume(left, right, num_disparities, is_right)


class ExportableCostVolume(nn.Module):

    def __init__(self, num_disparities, is_right):
        super().__init__()
        self.num_disparities = num_disparities
        self.is_right = is_right

    def forward(self, left, right):
        return ExportableCostVolumeFunction.apply(left, right, self.num_disparities, self.is_right)


@torch.autograd.function.traceable
class ExportableMatchabilityFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input):
        return g.op("trt::TRT_PluginV2", input, version_s="0.0.1", namespace_s="", data_s="", name_s="Matchability")

    @staticmethod
    def forward(ctx, input):
        return matchability(input)


class ExportableMatchability(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return ExportableMatchabilityFunction.apply(input)


@torch.autograd.function.traceable
class ExportableSoftArgminFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input):
        return g.op("trt::TRT_PluginV2", input, version_s="0.0.1", namespace_s="", data_s="", name_s="SoftArgmin")

    @staticmethod
    def forward(ctx, input):
        return soft_argmin(input)


class ExportableSoftArgmin(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return ExportableSoftArgminFunction.apply(input)


@torch.autograd.function.traceable
class ExportableFlattenFunction(torch.autograd.Function):

    def __init__(self):
        super().__init__()

    @staticmethod
    def symbolic(g, input, start_dim, end_dim):
        # Only support this for now.
        assert start_dim == 1
        assert end_dim == 2
        dim = input.type().dim()
        assert dim == 5
        sizes = input.type().sizes()
        new_shape = g.op("Constant",
                         value_t=torch.tensor([sizes[0], sizes[1] * sizes[2], sizes[3], sizes[4]], dtype=torch.int64))
        return g.op("Reshape", input, new_shape)

    @staticmethod
    def forward(ctx, input, start_dim, end_dim):
        return torch.flatten(input, start_dim, end_dim)


class ExportableFlatten(nn.Module):

    def __init__(self, start_dim, end_dim):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return ExportableFlattenFunction.apply(input, self.start_dim, self.end_dim)
