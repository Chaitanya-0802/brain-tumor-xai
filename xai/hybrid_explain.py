"""XAI for BrainHybrid.

Team contract:
    explain(model, img_tensor, target_class=None) -> np.ndarray float32 (128,128), in [0,1]

method="rollout" (default): attention rollout over the Transformer part (16x16 token grid)
method="gradcam": Grad-CAM on the last ReLU of the CNN stem (also 16x16)
"""
import numpy as np
import torch
import torch.nn.functional as F

import config
from xai.gradcam import GradCAM


def _normalize01(cam):
    cam = cam - cam.min()
    if cam.max() > 1e-8:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def attention_rollout(model, img_tensor):
    """Attention rollout (Abnar & Zuidema, 2020). Returns (heatmap, predicted_class, probs)."""
    model.eval()
    device = next(model.parameters()).device
    x = img_tensor.to(device)
    model.set_save_attn(True)                   # blocks only store attention when this is on
    try:
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(logits.argmax(1).item())

            attns = model.get_attentions()      # each (1, heads, N, N)
            n = attns[0].shape[-1]
            eye = torch.eye(n, device=device)
            rollout = eye.clone()
            for a in attns:
                a = a.mean(dim=1)[0]            # average over heads -> (N, N)
                a = 0.5 * a + 0.5 * eye         # account for the residual connection
                a = a / a.sum(dim=-1, keepdim=True)
                rollout = a @ rollout           # chain the layers

            mask = rollout[0, 1:]               # CLS row, drop the CLS column -> 256 values
            g = model.grid
            mask = mask.reshape(1, 1, g, g)
            mask = F.interpolate(mask, size=(config.IMG_SIZE, config.IMG_SIZE),
                                 mode="bilinear", align_corners=False)
            cam = mask[0, 0].cpu().numpy()
    finally:
        model.set_save_attn(False)              # switch off again so inference stays lean
    return _normalize01(cam), pred, probs


def gradcam_stem(model, img_tensor, target_class=None):
    """Grad-CAM on the CNN stem: last ReLU of the last ConvBlock (128 x 16 x 16)."""
    layer = model.stem[-1].block[5]
    engine = GradCAM(model, target_layer=layer)
    try:
        return engine(img_tensor, target_class)     # (heatmap, pred, probs)
    finally:
        engine.remove()


def explain(model, img_tensor, target_class=None, method="rollout"):
    """Contract function. Rollout is class-agnostic, so target_class is ignored for it."""
    if method == "rollout":
        heatmap, _, _ = attention_rollout(model, img_tensor)
    elif method == "gradcam":
        heatmap, _, _ = gradcam_stem(model, img_tensor, target_class)
    else:
        raise ValueError("method must be 'rollout' or 'gradcam'")
    return heatmap