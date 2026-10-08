"""Hybrid CNN-Transformer for brain tumor MRI classification (written from scratch).

Pipeline:
    image (B,3,128,128)
      -> CNN stem (3 ConvBlocks from models/cnn.py)       -> (B,128,16,16)
      -> 1x1 conv projection to embed_dim                  -> (B,192,16,16)
      -> flatten to 256 tokens                             -> (B,256,192)
      -> prepend CLS token + learnable positional embedding -> (B,257,192)
      -> 4 Transformer encoder blocks (Block from models/vit.py, pre-norm)
      -> final LayerNorm, CLS token, linear head           -> logits (B,4)

Team contract: input (B,3,128,128), output raw logits (B,4), no inplace ops.
Depends on: models/cnn.py (ConvBlock) and models/vit.py (Block) - do not rename them.
"""
import torch
import torch.nn as nn

import config
from models.cnn import ConvBlock   # Member 1: conv-BN-ReLU x2 + MaxPool (non-inplace)
from models.vit import Block       # Member 2: pre-norm encoder block with hand-written attention


class BrainHybrid(nn.Module):
    """Input (B,3,128,128) -> logits (B,4)."""

    def __init__(self, num_classes=4, embed_dim=192, depth=4, heads=3,
                 mlp_ratio=4, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        # 1) CNN stem: spatial 128 -> 64 -> 32 -> 16, channels 3 -> 32 -> 64 -> 128
        self.stem = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        # 2) 1x1 conv = the same linear layer applied to the 128-d vector at every position
        self.proj = nn.Conv2d(128, embed_dim, kernel_size=1)

        self.grid = config.IMG_SIZE // 8        # 16 (three MaxPool(2) layers)
        num_tokens = self.grid ** 2             # 256

        # 3) CLS token + learnable positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # 4) Transformer encoder (shared Block class from the ViT)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, heads, mlp_ratio, dropout, attn_dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 5) classification head on the CLS token
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for blk in self.blocks:                 # only the Transformer part, not the CNN stem
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        # LayerNorm defaults (weight=1, bias=0) are already correct.
        # The stem keeps PyTorch's default init, which suits conv + ReLU.

    def set_save_attn(self, flag: bool = True):
        """Turn attention saving on/off in every block (needed for attention rollout)."""
        for blk in self.blocks:
            blk.attn.save_attn = flag
            if not flag:
                blk.attn.attn_weights = None

    def get_attentions(self):
        """List (one per block) of (B, heads, 257, 257) post-softmax maps from the last forward().
        Only filled if set_save_attn(True) was called before forward()."""
        return [blk.attn.attn_weights for blk in self.blocks]

    def forward(self, x):
        B = x.shape[0]
        f = self.stem(x)                        # (B, 128, 16, 16)
        t = self.proj(f)                        # (B, 192, 16, 16)
        t = t.flatten(2).transpose(1, 2)        # (B, 256, 192): one token per grid cell
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 192)
        t = torch.cat([cls, t], dim=1)          # (B, 257, 192)
        t = self.pos_drop(t + self.pos_embed)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return self.head(t[:, 0])               # logits from the CLS token