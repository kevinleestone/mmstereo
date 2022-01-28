# Copyright 2021 Toyota Research Institute.  All rights reserved.

from importlib.machinery import SourceFileLoader
import os
import sys
import tempfile

from args import ModelConfig
import torch

from init.default_init import default_init

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))


def load_model(model_config: ModelConfig):
    # Load model module based on hparams using importlib and instantiate model.
    net_module = SourceFileLoader(model_config.model_name, model_config.model_file).load_module()
    net_attr = getattr(net_module, model_config.model_name)
    model = net_attr(model_config)

    # Initialize network with random weights.
    model.apply(default_init)

    if model_config.checkpoint is not None:
        state_dict = torch.load(model_config.checkpoint, map_location='cpu')['state_dict']
        keys = sorted(state_dict.keys())

        # The PyTorch Lightning checkpoints have a prefix of "model.", so strip that out before loading.
        for key in keys:
            prefix = "model."
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)

    return model


def get_cpu_model_copy(model_config: ModelConfig, model):
    exportable_model = load_model(model_config)
    with tempfile.TemporaryFile() as temp_state_dict_file:
        torch.save(model.state_dict(), temp_state_dict_file)
        temp_state_dict_file.seek(0)
        cpu_state_dict = torch.load(temp_state_dict_file, map_location="cpu")
    exportable_model.load_state_dict(cpu_state_dict)
    return exportable_model
