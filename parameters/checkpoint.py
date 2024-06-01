import json
import os

import numpy as np
import tensorflow as tf

from config import Context
from util.download import download_all

components = [
    "checkpoint",
    "encoder.json", "hparams.json", "model.ckpt.data-00000-of-00001",
    "model.ckpt.index", "model.ckpt.meta", "vocab.bpe"
]


def load_checkpoint(model_dir):
    ckpt_dir = tf.train.latest_checkpoint(model_dir)
    cfg = Context(json.load(open(os.path.join(model_dir, "hparams.json"))))

    params = {
        "blocks": [{} for _ in range(cfg.n_layer)]
    }

    for var_name, _ in tf.train.list_variables(ckpt_dir):
        var_array = np.squeeze(tf.train.load_variable(ckpt_dir, var_name))
        var_path = var_name.split("/")[1:]
        target = params
        if var_path[0].startswith("h"):
            layer = int(var_path[0][1:])
            target = params["blocks"][layer]

        for key in var_path[1:-1]:
            target = target.setdefault(key, {})

        last = var_path[-1]
        target[last] = var_array

    return cfg, Context(params)


def download_checkpoint(base, label, models_dir):
    model_dir = os.path.join(models_dir, label)
    download_all(base, components, model_dir, label)
    return model_dir
