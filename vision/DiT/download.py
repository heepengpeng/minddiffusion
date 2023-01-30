# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
import logging
import os

import mindspore as ms
from mindcv.utils.download import DownLoad

pretrained_models = {'DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt'}


def find_model(model_name):
    """
    Finds a pre-trained G.pt model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained G.pt checkpoints
        return download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        return ms.load_checkpoint(model_name)


def torch_to_mindspore(state_dict):
    ms_ckpt = []
    for k, v in state_dict.items():
        if 'embedding_table' in k:
            k = k.replace('weight', 'embedding_table')
        ms_ckpt.append({'name': k, 'data': ms.Tensor(v.numpy())})
    return ms_ckpt


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        DownLoad().download_url(url=web_path, path='pretrained_models')
    try:
        import torch
    except:
        raise ImportError(f"'import torch' failed, please install torch by "
                          f"`pip install torch` or instructions from 'https://pytorch.org'")
    from mindspore.train.serialization import save_checkpoint
    logging.info('Starting checkpoint conversion.')
    state_dict = torch.load(local_path, map_location=torch.device('cpu'))
    ms_ckpt = torch_to_mindspore(state_dict)
    ms_ckpt_path = local_path.replace('.pt', '.ckpt')
    save_checkpoint(ms_ckpt, ms_ckpt_path)
    return ms.load_checkpoint(ms_ckpt_path)


if __name__ == "__main__":
    # Download all DiT checkpoints
    for model in pretrained_models:
        download_model(model)
    print('Done.')
