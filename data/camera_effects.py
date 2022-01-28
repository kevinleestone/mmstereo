# Copyright 2021 Toyota Research Institute.  All rights reserved.

"""Apply camera effects to simulated images

Portions of code borrowed from Mike/Kevin's stuff.
Camera processing pipeline references:
  https://web.stanford.edu/class/cs231m/lectures/lecture-11-camera-isp.pdf
  https://arxiv.org/pdf/1811.11127.pdf
"""

import random

import kornia
import torch
import torch.nn.functional as F

SKIP_PROB = 0.1
BAYER_PROB = 0.67
NOISE_PROB = 0.75
BLUR_PROB = 0.25
CHROM_PROB = 0.5


def compose(x, gen_shared=random.Random(), gen_private=random.Random()):
    torch_seed = gen_private.randint(0, 1 << 30)

    # # hack
    # gen_shared = gen_private

    with torch.random.fork_rng(devices=[x.device]):
        torch.random.manual_seed(torch_seed)

        do_skip = gen_shared.random() < SKIP_PROB
        if do_skip:
            return x

        do_bayer = gen_shared.random() < BAYER_PROB
        do_noise = gen_shared.random() < NOISE_PROB
        do_blur = gen_shared.random() < BLUR_PROB
        do_chrom = gen_shared.random() < CHROM_PROB

        x = srgb_to_linear(x)

        if do_blur:
            x = blur(x, gen_shared, gen_private)

        if do_chrom:
            x = chromatic_aberration_upsample(x, gen_shared, gen_private)

        if do_bayer:
            x, bayer_pattern = bayer(x, gen_shared, gen_private)

        if do_noise:
            x = shot_read_noise(x, gen_shared, gen_private)

            x = torch.clamp(x, 0., 1.)

        if do_bayer:
            x = debayer(x, bayer_pattern)

        x = linear_to_srgb(x)

        x = torch.clamp(x, 0., 1.)
        return x


def blur(x, gen_shared, gen_private):
    batch_size, _, _, _ = x.shape
    for i in range(batch_size):
        kernel_size = gen_shared.randint(1, 2) * 2 + 1
        blur_type = gen_shared.randint(0, 1)
        kernel_size_offset = gen_private.randint(-1, 1) * 2
        if gen_private.randint(0, 1) == 0:
            kernel_size_x = kernel_size + kernel_size_offset
            kernel_size_y = kernel_size
        else:
            kernel_size_x = kernel_size
            kernel_size_y = kernel_size + kernel_size_offset
        if blur_type == 0:
            sigma = gen_private.gauss(kernel_size / 2.5, kernel_size / 8)
            x[i:i + 1] = kornia.filters.gaussian_blur2d(x[i:i + 1], (kernel_size_y, kernel_size_x), (sigma, sigma))
        elif blur_type == 1:
            x[i:i + 1] = kornia.filters.box_blur(x[i:i + 1], (kernel_size_y, kernel_size_x))
    return x


def chromatic_aberration_upsample(x, gen_shared, gen_private):
    batch_size, _, height, width = x.shape

    stddev = 0.67
    for b in range(batch_size):
        i = gen_private.randint(0, 2)
        top = max(0, int(round(gen_shared.gauss(0.0, stddev))))
        bottom = max(0, int(round(gen_shared.gauss(0.0, stddev))))
        left = max(0, int(round(gen_shared.gauss(0.0, stddev))))
        right = max(0, int(round(gen_shared.gauss(0.0, stddev))))
        total_x = left + right
        total_y = top + bottom
        top_offset = gen_shared.randint(0, total_y) if total_y > 0 else 0
        left_offset = gen_shared.randint(0, total_x) if total_x > 0 else 0

        if total_x > 0 or total_y > 0:
            upsampled = F.interpolate(x[b:b + 1, i:i + 1, :, :], (height + total_y, width + total_x),
                                      mode="bilinear", align_corners=False)
            x[b:b + 1, i:i + 1, :, :] = upsampled[:, :, top_offset:top_offset + height,
                                        left_offset:left_offset + width]

    return x


def srgb_to_linear(image):
    return torch.pow(image, 2.2)


def make_noise(image, nscales):
    # Based on:
    #   https://arxiv.org/pdf/1811.11127.pdf (section 3.1)
    batch_size, _, height, width = image.shape
    min_noise = 0.2 / nscales
    max_noise = 0.5 / nscales
    a = torch.log10(torch.tensor(0.0001, dtype=image.dtype, device=image.device))
    b = torch.log10(torch.tensor(0.012, dtype=image.dtype, device=image.device))

    noise_level = torch.zeros((batch_size, 1, 1, 1), dtype=image.dtype, device=image.device).uniform_(min_noise,
                                                                                                      max_noise)

    log_lambda_shot = a * (1 - noise_level) + b * noise_level
    log_lambda_read = 2.18 * log_lambda_shot + 1.2
    var = 10 ** log_lambda_read + (10 ** log_lambda_shot) * image
    return torch.normal(0, torch.sqrt(var))


def shot_read_noise(x, gen_shared, gen_private):
    _, _, original_height, original_width = x.shape
    noise_type = gen_private.randint(0, 2)
    if noise_type == 0:
        return x + make_noise(x, 1)
    elif noise_type == 1:
        sx = gen_private.uniform(0.75, 1.0)
        sy = gen_private.uniform(0.75, 1.0)
        scaled = F.interpolate(x, scale_factor=(sx, sy), mode="nearest")
        noise = make_noise(scaled, 1)
        noise = F.interpolate(noise, size=(original_height, original_width), mode="bilinear", align_corners=False)

        return x + noise
    else:
        sx_high = gen_private.uniform(0.75, 1.0)
        sy_high = gen_private.uniform(0.75, 1.0)
        scaled_high = F.interpolate(x, scale_factor=(sx_high, sy_high), mode="nearest")
        noise_high = make_noise(scaled_high, 2)
        noise_high = F.interpolate(noise_high, size=(original_height, original_width), mode="bilinear",
                                   align_corners=False)

        sx_low = gen_private.uniform(0.25, 0.5)
        sy_low = gen_private.uniform(0.25, 0.5)
        scaled_low = F.interpolate(x, scale_factor=(sx_low, sy_low), mode="nearest")
        noise_low = make_noise(scaled_low, 2)
        noise_low = F.interpolate(noise_low, size=(original_height, original_width), mode="bilinear",
                                  align_corners=False)

        return x + noise_high + noise_low


def get_bayer_filter(x, pattern):
    _, _, height, width = x.shape
    assert height % 2 == 0
    assert width % 2 == 0

    def _get_tiling(pattern):
        assert pattern in ['RGGB', 'BGGR', 'GRBG', 'GBRG']
        pixels = {
            'R': torch.tensor([0., 0., 1.], dtype=x.dtype, device=x.device),
            'G': torch.tensor([0., 1., 0.], dtype=x.dtype, device=x.device),
            'B': torch.tensor([1., 0., 0.], dtype=x.dtype, device=x.device),
        }
        return torch.cat([pixels[p] for p in pattern]).reshape([2, 2, 3]).permute((2, 0, 1)).unsqueeze(0)

    tiling = _get_tiling(pattern)
    filter_ = tiling.repeat((1, 1, height // 2, width // 2))
    return filter_


def bayer(x, gen_shared, gen_private):
    pattern = gen_shared.choice(['BGGR', 'RGGB', 'GBRG', 'GRBG'])
    filter_ = get_bayer_filter(x, pattern=pattern)
    x = torch.sum(x * filter_, dim=1, keepdim=True)
    return x, pattern


def debayer(x, pattern):
    # Adapted from colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004.

    filter_ = get_bayer_filter(x, pattern=pattern)

    GR_GB = torch.tensor(
            [[[[0, 0, -1, 0, 0],
               [0, 0, 2, 0, 0],
               [-1, 2, 4, 2, -1],
               [0, 0, 2, 0, 0],
               [0, 0, -1, 0, 0]]]], dtype=x.dtype, device=x.device) / 8

    Rg_RB_Bg_BR = torch.tensor(
            [[[[0, 0, 0.5, 0, 0],
               [0, -1, 0, -1, 0],
               [-1, 4, 5, 4, - 1],
               [0, -1, 0, -1, 0],
               [0, 0, 0.5, 0, 0]]]], dtype=x.dtype, device=x.device) / 8

    Rg_BR_Bg_RB = torch.transpose(Rg_RB_Bg_BR, 2, 3)

    Rb_BB_Br_RR = torch.tensor(
            [[[[0, 0, -1.5, 0, 0],
               [0, 2, 0, 2, 0],
               [-1.5, 0, 6, 0, -1.5],
               [0, 2, 0, 2, 0],
               [0, 0, -1.5, 0, 0]]]], dtype=x.dtype, device=x.device) / 8

    R_m = filter_[:, 2:3, :, :]
    G_m = filter_[:, 1:2, :, :]
    B_m = filter_[:, 0:1, :, :]

    R = x * R_m
    G = x * G_m
    B = x * B_m

    x_padded = F.pad(x, (2, 2, 2, 2), mode='reflect')
    cfa_gr_gb = F.conv2d(x_padded, GR_GB).to(x.dtype)
    G = torch.where((R_m == 1) + (B_m == 1), cfa_gr_gb, G)

    RBg_RBBR = F.conv2d(x_padded, Rg_RB_Bg_BR).to(x.dtype)
    RBg_BRRB = F.conv2d(x_padded, Rg_BR_Bg_RB).to(x.dtype)
    RBgr_BBRR = F.conv2d(x_padded, Rb_BB_Br_RR).to(x.dtype)

    # Red rows.
    R_r = torch.any(R_m == 1, dim=3, keepdim=True).expand(-1, -1, -1, R.shape[3])
    # Red columns.
    R_c = torch.any(R_m == 1, dim=2, keepdim=True).expand(-1, -1, R.shape[2], -1)
    # Blue rows.
    B_r = torch.any(B_m == 1, dim=3, keepdim=True).expand(-1, -1, -1, B.shape[3])
    # Blue columns
    B_c = torch.any(B_m == 1, dim=2, keepdim=True).expand(-1, -1, B.shape[2], -1)

    R = torch.where(R_r * B_c, RBg_RBBR, R)
    R = torch.where(B_r * R_c, RBg_BRRB, R)

    B = torch.where(B_r * R_c, RBg_RBBR, B)
    B = torch.where(R_r * B_c, RBg_BRRB, B)

    R = torch.where(B_r * B_c, RBgr_BBRR, R)
    B = torch.where(R_r * R_c, RBgr_BBRR, B)

    out = torch.cat([B, G, R], dim=1)

    return out


def linear_to_srgb(x, epsilon=1e-8):
    x = torch.max(x, torch.tensor([epsilon], dtype=x.dtype, device=x.device))
    x = x.pow(1 / 2.2)
    return x
