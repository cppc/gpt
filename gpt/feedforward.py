from torch import nn

from gpt.activation import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim)
        )

    def forward(self, x):
        out_x = self.layers(x)
        return out_x
