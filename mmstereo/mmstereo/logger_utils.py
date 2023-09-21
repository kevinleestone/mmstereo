# Copyright 2021 Toyota Research Institute.  All rights reserved.

import os

import pytorch_lightning as pl


def get_rank():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0


def get_loggers(hparams):
    root = hparams.output
    os.makedirs(root, exist_ok=True)

    # Add optional note to our output path if set.
    if hparams.note is not None:
        run_name = hparams.note
    else:
        run_name = "default"
    run_path = os.path.join(root, run_name)

    if get_rank() == 0:
        os.makedirs(run_path, exist_ok=True)

        tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=root, name=run_name)
        loggers = [tensorboard_logger]

        checkpoint_dir = os.path.join(tensorboard_logger.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=False)
    else:
        loggers = True
        checkpoint_dir = "/tmp"

    return loggers, run_path, checkpoint_dir


def get_tensorboard(loggers):
    for logger in loggers:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            return logger
    return None
