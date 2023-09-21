# Copyright 2021 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as F


def null_loss():
    return None, False


def dummy_loss(tensor):
    tensor[torch.isnan(tensor)] = 0.0
    return F.mse_loss(tensor, torch.zeros_like(tensor)) * 0.0, False


def valid_loss(tensor):
    if not torch.any(torch.isnan(tensor)):
        return tensor, True
    else:
        return dummy_loss(tensor)
