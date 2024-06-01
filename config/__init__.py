from .context import Context

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True  # Query-Key-Value bias
}

model_configs = {
    "gpt2-test": {
        "label": "test",
        "config": {
            "context_length": 256
        }
    },
    "gpt2-small": {
        "label": "124M",
        "config": {

        }
    },
    "gpt2-medium": {
        "label": "355M",
        "config": {
            "emb_dim": 1024,
            "n_layers": 24,
            "num_heads": 16
        }
    },
    "gpt2-large": {
        "label": "774M",
        "config": {
            "emb_dim": 1280,
            "n_layers": 36,
            "num_heads": 20
        }
    },
    "gpt2-xlarge": {
        "label": "774M",
        "config": {
            "emb_dim": 1600,
            "n_layers": 48,
            "num_heads": 25
        }
    }
}


def gpt2_config(size):
    model = model_configs[f"gpt2-{size}"]
    return Context(
        GPT_CONFIG_124M,
        model["config"]
    ), model["label"]
