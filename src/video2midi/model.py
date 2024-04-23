# Inspired by: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vivit.py

import torch
from torch import nn
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# Helper functions
def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# FeedForward module
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Attention module
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer module
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_frames,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool='mean',  
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b f c h w -> b f (h w c)'),  
            nn.Linear(channels * image_size[0] * image_size[1], dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool 
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        # Apply pooling across the temporal dimension
        if self.pool == 'mean':
            x = x.mean(dim=1, keepdim=True)  # Mean pooling
        else:
            x = x.max(dim=1, keepdim=True)[0]  # Max pooling

        # Expand the pooled representation to match the number of frames
        x = x.expand(-1, video.shape[1], -1)

        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x
