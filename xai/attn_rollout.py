"""Attention Rollout (Abnar & Zuidema, 2020). Class-agnostic: target_class is accepted
for interface compatibility. Defaults (head_fusion='mean', discard_ratio=0.0) give the
standard method required by the team contract."""
import numpy as np
import torch
import torch.nn.functional as F

import config


@torch.no_grad()
def explain(model, img_tensor, target_class=None, head_fusion="mean", discard_ratio=0.0):
    """img_tensor: (1,3,H,W) or (3,H,W), normalized. Returns (128,128) float32 in [0,1]."""
    was_training = model.training
    model.eval()
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    model.set_save_attn(True)
    try:
        model(img_tensor)
        atts = model.get_attentions()
    finally:
        model.set_save_attn(False)
        model.train(was_training)

    rollout = None
    for a in atts:
        a = a[0]                                              # (heads, N, N)
        if head_fusion == "mean":
            a = a.mean(dim=0)
        elif head_fusion == "max":
            a = a.amax(dim=0)
        elif head_fusion == "min":
            a = a.amin(dim=0)
        else:
            raise ValueError("head_fusion must be 'mean', 'max' or 'min'")
        if discard_ratio > 0:                                 # drop the weakest attention values
            thr = torch.quantile(a.flatten(), discard_ratio)
            a = torch.where(a < thr, torch.zeros_like(a), a)
        a = a + torch.eye(a.size(0), device=a.device)         # residual connection
        a = a / a.sum(dim=-1, keepdim=True)                   # re-normalize rows
        rollout = a if rollout is None else a @ rollout       # later layer on the left

    mask = rollout[0, 1:]                                     # CLS row, drop CLS entry
    g = int(round(mask.numel() ** 0.5))
    mask = mask.reshape(1, 1, g, g)
    mask = F.interpolate(mask, size=(config.IMG_SIZE, config.IMG_SIZE),
                         mode="bilinear", align_corners=False)
    m = mask[0, 0].float().cpu().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.astype(np.float32)
