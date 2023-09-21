# Copyright 2021 Toyota Research Institute.  All rights reserved.

import argparse
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import visualization

VIS_DISPARITY = 256
FX = 1075.0
FY = 1220.0


def run_inference(model, left_file, right_file):
    left = cv2.imread(left_file)
    right = cv2.imread(right_file)

    # Convert inputs from Numpy arrays scaled 0 to 255 to PyTorch tensors scaled from 0 to 1.
    left_tensor = left.astype(np.float32).transpose((2, 0, 1)) / 255.0
    right_tensor = right.astype(np.float32).transpose((2, 0, 1)) / 255.0
    left_tensor = torch.from_numpy(left_tensor).unsqueeze(0)
    right_tensor = torch.from_numpy(right_tensor).unsqueeze(0)

    # Crop inputs such that they don't need any padding when passing to the network.5)
    height, width, _ = left.shape
    target_height = int(math.ceil(height / 16) * 16)
    target_width = int(math.ceil(width / 16) * 16)
    padding_x = target_width - width
    padding_y = target_height - height
    left_tensor = F.pad(left_tensor, (0, padding_x, 0, padding_y))
    right_tensor = F.pad(right_tensor, (0, padding_x, 0, padding_y))

    # Move model and inputs to GPU.
    model.cuda()
    model.eval()
    left_tensor = left_tensor.cuda()
    right_tensor = right_tensor.cuda()

    # Do forward pass on model and get output.
    with torch.no_grad():
        output, all_outputs = model(left_tensor, right_tensor)
    disparity = output["disparity"]
    disparity_small = output["disparity_small"]
    matchability = output.get("matchability", None)

    scale = disparity.shape[3] // disparity_small.shape[3]

    # Generate visualizations for network output.
    disparity_vis = visualization.make_cv_disparity_image(
        disparity[0, 0, :, :], VIS_DISPARITY
    )
    disparity_small_vis = visualization.make_cv_disparity_image(
        disparity_small[0, 0, :, :], VIS_DISPARITY // scale
    )
    disparity_small_vis = cv2.resize(
        disparity_small_vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    confidence_vis = visualization.make_cv_confidence_image(
        torch.exp(matchability[0, 0, :, :])
    )
    confidence_vis = cv2.resize(
        confidence_vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )

    disparity_vis = disparity_vis[:height, :width, :]
    disparity_small_vis = disparity_small_vis[:height, :width, :]
    confidence_vis = confidence_vis[:height, :width, :]

    # Put all the visualizations together and display in a window.
    vis_top = cv2.hconcat([left, disparity_vis])
    vis_bottom = cv2.hconcat([confidence_vis, disparity_small_vis])
    vis = cv2.vconcat([vis_top, vis_bottom])

    cv2.imshow("vis", vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Torchscript file produced by training",
    )
    parser.add_argument(
        "--left",
        default="left.png",
        type=str,
        help="Filename of left image to use for inference",
    )
    parser.add_argument(
        "--right",
        default="right.png",
        type=str,
        help="Filename of right image to use for inference",
    )
    hparams = parser.parse_args()

    script = torch.jit.load(hparams.script)

    run_inference(script, hparams.left, hparams.right)
