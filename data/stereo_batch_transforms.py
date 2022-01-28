# Copyright 2021 Toyota Research Institute.  All rights reserved.

import random

from data.sample import ElementKeys
from data.camera_effects import compose


class CameraEffect(object):
    """Applies random camera effects to simulated images"""

    def __init__(self, generator):
        self.generator = generator

    def __call__(self, batch):
        batch_metadata, batch_data = batch
        if self.generator is not None:
            # Unpack data from the batch.
            sim = batch_metadata.simulated
            left_image = batch_data[ElementKeys.LEFT_RGB]
            right_image = batch_data[ElementKeys.RIGHT_RGB]
            batch_size, _, _, _ = left_image.shape
            for idx in range(batch_size):
                # Only apply this operation to simulated images.
                if sim[idx].item():
                    shared_seed = self.generator.randint(0, 1000000)

                    # Left and right are augmented both a shared generator where both will receive the exact same
                    # random values and a private generator where each will get unique values.
                    left_image[idx:idx + 1, :, :, :] = compose(left_image[idx:idx + 1, :, :, :],
                                                               random.Random(shared_seed), self.generator)
                    right_image[idx:idx + 1, :, :, :] = compose(right_image[idx:idx + 1, :, :, :],
                                                                random.Random(shared_seed), self.generator)
            # Pack data back into the batch.
            batch_data[ElementKeys.LEFT_RGB] = left_image
            batch_data[ElementKeys.RIGHT_RGB] = right_image
        return batch_metadata, batch_data
