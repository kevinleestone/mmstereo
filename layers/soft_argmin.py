# Copyright 2021 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def soft_argmin(input):
    _, channels, _, _ = input.shape

    softmin = F.softmin(input, dim=1)
    index_tensor = torch.arange(0, channels, dtype=softmin.dtype, device=softmin.device).view(1, channels, 1, 1)
    output = torch.sum(softmin * index_tensor, dim=1, keepdim=True)
    return output


class SoftArgmin(nn.Module):
    """Compute soft argmin operation for given cost volume"""

    def forward(self, input):
        if torch.jit.is_scripting():
            # Torchscript generation can't handle mixed precision, so always compute at float32.
            return soft_argmin(input)
        else:
            return self.forward_with_amp(input)

    @torch.jit.unused
    def forward_with_amp(self, input):
        """This operation is unstable at float16, so compute at float32 even when using mixed precision"""
        with torch.cuda.amp.autocast(enabled=False):
            input = input.to(torch.float32)
            return soft_argmin(input)
