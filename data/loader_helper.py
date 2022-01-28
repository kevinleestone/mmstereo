# Copyright 2021 Toyota Research Institute.  All rights reserved.

import os

import cv2
import numpy as np
from turbojpeg import TurboJPEG


class LoaderHelper(object):
    """Helper for loading data elements of various file types with optional fallbacks if the file doesn't exist"""

    def __init__(self):
        self.jpeg = TurboJPEG()

    def load_image(self, filename, *, fallback_example=None, fallback_none=False):
        if os.path.exists(filename):
            ext = os.path.splitext(filename[1])
            if "jpg" in ext:
                with open(filename, "rb") as jpeg_file:
                    return self.jpeg.decode(jpeg_file.read())
            else:
                return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        else:
            if fallback_example is not None:
                return np.zeros_like(fallback_example)
            else:
                if fallback_none:
                    return None
                else:
                    assert False

    def load_npz(self, filename, *, fallback_example=None, fallback_none=False):
        if os.path.exists(filename):
            with np.load(filename) as npz:
                return npz["arr_0"].astype(np.float32)
        else:
            if fallback_example is not None:
                return np.zeros_like(fallback_example)
            else:
                if fallback_none:
                    return None
                else:
                    assert False
