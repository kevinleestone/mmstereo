# Copyright 2021 Toyota Research Institute.  All rights reserved.

from multiprocessing import Pool
import os
import shutil

import cv2
import numpy as np


IN = "datasets/sceneflow/raw"
OUT = "datasets/sceneflow/flying_things"


def handle_files(args):
    infile, outfile = args

    if ".png" in infile:
        shutil.copy(infile, outfile)
    else:
        disparity = -cv2.imread(infile, cv2.IMREAD_UNCHANGED)
        with open(outfile, "wb") as out_file:
            np.savez_compressed(out_file, disparity)


def main():
    files = []
    for root, dirnames, filenames in os.walk(IN):
        for filename in filenames:
            split_dir = "train" if "/train/" in root else "val"
            camera = "left" if "/left" in root else "right"

            if ".png" in filename:
                out_dir = os.path.join(OUT, split_dir, camera)
                os.makedirs(out_dir, exist_ok=True)

                files.append(
                    (os.path.join(root, filename), os.path.join(out_dir, filename))
                )
            elif ".pfm" in filename:
                out_dir = os.path.join(OUT, split_dir, camera + "_disparity")
                os.makedirs(out_dir, exist_ok=True)

                files.append(
                    (
                        os.path.join(root, filename),
                        os.path.join(out_dir, filename.replace(".pfm", ".npz")),
                    )
                )

    pool = Pool(processes=16)
    pool.map(handle_files, files)


if __name__ == "__main__":
    main()
