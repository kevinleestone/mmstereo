# Copyright 2021 Toyota Research Institute.  All rights reserved.

import argparse
import warnings

warnings.filterwarnings(
    "ignore",
    "The default behavior for interpolate/upsample with float scale_factor changed ",
)
warnings.filterwarnings(
    "ignore",
    "torch.tensor results are registered as constants in the trace. You can safely ",
)
warnings.filterwarnings(
    "ignore", "Default grid_sample and affine_grid behavior has changed "
)
warnings.filterwarnings(
    "ignore",
    "Named tensors and all their associated APIs are an experimental feature and ",
)

from omegaconf import OmegaConf
import pytorch_lightning as pl

from args import TrainingConfig
from callbacks import OnnxExport, TorchscriptExport
from data.stereo_data_module import StereoDataModule
from logger_utils import get_loggers
from model import StereoModel

if __name__ == "__main__":
    # This seeds all random sources, including Python , Numpy, and PyTorch CPU and GPU.
    pl.utilities.seed.seed_everything(12345)

    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--override", type=str, nargs="+")
    args = parser.parse_args()

    if args.override is not None and len(args.override) > 0:
        override_config = OmegaConf.from_cli(args.override)
    else:
        override_config = {}

    yaml_config = OmegaConf.load(args.config)
    hparams: TrainingConfig = OmegaConf.merge(
        TrainingConfig, yaml_config, override_config
    )
    if args.print_config:
        print(OmegaConf.to_yaml(hparams))
    if args.scratch:
        hparams.output = "scratch"

    # Mixed precision training uses 16-bit precision floats, otherwise use 32-bit floats.
    precision = 16 if hparams.use_amp else 32

    stereo_model = StereoModel(hparams)
    data_module = StereoDataModule(hparams.data)

    loggers, root_dir, checkpoint_dir = get_loggers(hparams)

    # Callbacks to create model checkpoints and artifacts.
    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        monitor="val_loss",
        save_top_k=hparams.epochs // 5 if hparams.epochs > 50 else 5,
    )

    # Callback to log learning rate.
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpointer, lr_monitor]
    if hparams.export_onnx:
        callbacks.append(OnnxExport(checkpoint_dir, hparams.note))
    if hparams.export_torchscript:
        callbacks.append(TorchscriptExport(checkpoint_dir, hparams.note))

    if hparams.distributed:
        gpus = -1
        accelerator = "ddp"
    else:
        gpus = [0]
        accelerator = None

    log_every_n_steps = 50

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        max_epochs=hparams.epochs,
        gpus=gpus,
        default_root_dir=root_dir,
        deterministic=False,
        precision=precision,
        accelerator=accelerator,
        log_every_n_steps=log_every_n_steps,
    )

    trainer.fit(stereo_model, data_module)
