from torch import nn

from attention.multihead_attention import MultiHeadAttention
from gpt.feedforward import FeedForward
from gpt.normalize import LayerNorm


def shortcut(x, norm, stage, drop):
    return x + drop(stage(norm(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_resid = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        scut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + scut  # Add the original input back

        # Shortcut connection for feed-forward block
        scut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + scut  # Add the original input back
        # x = shortcut(
        #     shortcut(x,
        #              self.norm1,
        #              self.att,
        #              self.drop_resid),
        #     self.norm2,
        #     self.ff,
        #     self.drop_resid)

        return x
