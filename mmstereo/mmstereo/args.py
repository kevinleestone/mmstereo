# Copyright 2021 Toyota Research Institute.  All rights reserved.

from dataclasses import dataclass, field
from typing import Optional, List

from omegaconf import MISSING


@dataclass
class DatasetConfig(object):
    """Path where dataset can be found"""

    path: str = MISSING

    """True if dataset is simulated"""
    sim: bool = False
    """Fraction of dataset to load"""
    fraction: float = 1.0
    """Number of times a dataset should be repeated per epoch"""
    repeats: int = 1

    no_split: bool = False


@dataclass
class TransformConfig(object):
    """Random crop images, individual arguments are height and width"""

    random_crop: Optional[List[int]] = None

    """Random horizontal flip"""
    random_horizontal_flip: bool = False

    """Random augmentation of image hue, saturation, gamma, intensity"""
    random_color_jitter: bool = False

    """Keep uncorrupted RGB images"""
    keep_uncorrupted: bool = False


@dataclass
class StageConfig(object):
    datasets: List[DatasetConfig] = field(default_factory=lambda: [])
    transform: TransformConfig = TransformConfig()

    """Shuffle batches during loading"""
    shuffle: bool = False
    """Batch size"""
    batch_size: int = MISSING
    """Number of processes to use for data loading"""
    num_workers: int = 4


@dataclass
class DataConfig(object):
    train: StageConfig = StageConfig()
    val: StageConfig = StageConfig()


@dataclass
class ModelConfig(object):
    model_file: str = MISSING
    model_name: str = MISSING

    fe_features: int = 16
    fe_internal_features: int = 32
    num_disparities: int = 256
    downsample_factor: int = 8

    checkpoint: Optional[str] = None


@dataclass
class LossConfig(object):
    """Multiplier applied to the disparity loss when calculating the total loss"""

    disparity_mult: float = 1.0
    """If true, the loss will be scaled based on the standard deviation and mean of the ground truth disparities"""
    disparity_stdmean_scaled: bool = False

    nsce_mult: float = 0.0
    smoothness_mult: float = 0.0


@dataclass
class OptimizerConfig(object):
    """Type of optimization, either SGD or Adam"""

    optimizer: str = "sgd"
    """Initial learning rate"""
    learning_rate: float = 0.02
    """Momentum when using SGD optimizer"""
    momentum: float = 0.9

    """L2 weight decay applied by the optimizer"""
    weight_decay: float = 1e-4

    """Learning rate scheduling policy, either none or poly"""
    lr_policy: str = "poly"
    """Exponent for polynomial learning rate decay"""
    poly_exp: float = 0.9


@dataclass
class TrainingConfig(object):
    data: DataConfig = DataConfig()

    model: ModelConfig = ModelConfig()

    loss: LossConfig = LossConfig()
    optimizer: OptimizerConfig = OptimizerConfig()

    """Number of epochs for training"""
    epochs: int = MISSING
    """Directory where training output such as logs and checkpoints should be stored"""
    output: str = MISSING
    """Note use to organize output in the output directory"""
    note: Optional[str] = None

    """Use automatic mixed precision during training, allowing for larger batch sizes and crops"""
    use_amp: bool = False
    """Automatically export ONNX at end of validation each epoch, stored in the checkpoints directory"""
    export_onnx: bool = False
    """Automatically export Torchscript at end of validation each epoch, stored in the checkpoints directory"""
    export_torchscript: bool = False
    """Train with multiple gpus using distributed data parallel"""
    distributed: bool = False
    sync_bn: bool = False

    """Apply random camera effects to simulated cameras such as noise and chromatic aberration"""
    random_camera_effect: bool = False
