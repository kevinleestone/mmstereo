# Copyright 2021 Toyota Research Institute.  All rights reserved.
#
# Originally from Koichiro Yamaguchi's pixwislab repo.

import torch.nn as nn


def default_init(module):
    """Initialize parameters of the module.
    """
    std = 0.01
    mode = "fan_in"
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode=mode, nonlinearity='relu')
        if module.bias is not None:
            nn.init.normal_(module.bias.data, mean=0.0, std=std)
    elif isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight.data, mode=mode, nonlinearity='relu')
        if module.bias is not None:
            nn.init.normal_(module.bias.data, mean=0.0, std=std)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, mean=1.0, std=std)
        nn.init.normal_(module.bias.data, mean=0.0, std=std)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.normal_(module.weight.data, mean=1.0, std=std)
        nn.init.normal_(module.bias.data, mean=0.0, std=std)
    elif isinstance(module, nn.GroupNorm):
        nn.init.normal_(module.weight.data, mean=1.0, std=std)
        nn.init.normal_(module.bias.data, mean=0.0, std=std)
