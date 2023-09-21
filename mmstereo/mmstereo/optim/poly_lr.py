# Copyright 2021 Toyota Research Institute.  All rights reserved.
#
# Originally from Koichiro Yamaguchi's pixwislab repo.


def lambda_poly_lr(max_epochs, exponent):
    """Make a function for computing learning rate by "poly" policy.

    This policy does a polynomial decay of the learning rate over the epochs
    of training.

    Args:
        max_epochs (int): max numbers of epochs
        exponent (float): exponent value
    """
    return lambda epoch: pow((1.0 - epoch / max_epochs), exponent)
