"""BrainViT: Vision Transformer written from scratch (pre-norm encoder, hand-written attention)."""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split the image into non-overlapping patches and project each to `dim`.
    A Conv2d with kernel = stride = patch_size is exactly 'flatten each patch + shared linear layer'."""

    def __init__(self, img_size=128, patch_size=8, in_chans=3, dim=192):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                      # (B,3,128,128)
        x = self.proj(x)                       # (B,dim,8,8)
        return x.flatten(2).transpose(1, 2)    # (B,64,dim)


class Attention(nn.Module):
    """Multi-head self-attention with explicit Q/K/V. Stores post-softmax weights when save_attn=True."""

    def __init__(self, dim, heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.save_attn = False
        self.attn_weights = None               # (B, heads, N, N) after softmax

    def forward(self, x):
        B, N, C = x.shape
        H, D = self.heads, self.head_dim
        q = self.q(x).view(B, N, H, D).transpose(1, 2)   # (B,H,N,D)
        k = self.k(x).view(B, N, H, D).transpose(1, 2)
        v = self.v(x).view(B, N, H, D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B,H,N,N)  QK^T / sqrt(d)
        attn = attn.softmax(dim=-1)
        if self.save_attn:
            self.attn_weights = attn.detach()
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # concat heads
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, dim, ratio=4, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * ratio, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim, heads, mlp_ratio=4, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BrainViT(nn.Module):
    def __init__(self, num_classes=4, img_size=128, patch_size=8, in_chans=3,
                 dim=192, depth=6, heads=3, mlp_ratio=4, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp_ratio, dropout, attn_dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def set_save_attn(self, flag: bool = True):
        for blk in self.blocks:
            blk.attn.save_attn = flag
            if not flag:
                blk.attn.attn_weights = None

    def get_attentions(self):
        """List (one per layer) of (B, heads, N, N) post-softmax attention maps."""
        return [blk.attn.attn_weights for blk in self.blocks]

    def forward(self, x):
        x = self.patch_embed(x)                                  # (B,64,dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)                           # (B,65,dim)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                                # CLS token -> logits
