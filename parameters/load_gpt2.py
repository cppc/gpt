import os

from parameters.checkpoint import download_checkpoint, load_checkpoint

allowed_labels = ("124M", "355M", "774M", "1558M")
base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"


def load_gpt2(label, models_dir):
    if label not in allowed_labels:
        raise ValueError(f"Model {label} not in {allowed_labels}")

    model_dir = download_checkpoint(base_url, label, models_dir)
    settings, params = load_checkpoint(model_dir)
    return settings, params