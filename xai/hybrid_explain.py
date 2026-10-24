"""XAI for BrainHybrid (Member 3).

Team contract:
    explain(model, img_tensor, target_class=None) -> np.ndarray float32 (128, 128), values in [0, 1]

Two methods:
    "rollout"  (default) attention rollout over the transformer part, on the 16x16 token grid
    "gradcam"  Grad-CAM on the last ReLU of the CNN stem (reuses GradCAM from xai/gradcam.py)

Works with the repo's models/hybrid.py: attention is only stored after model.set_save_attn(True).
"""
import numpy as np
import torch
import torch.nn.functional as F

import config
from xai.gradcam import GradCAM


def _normalize_map(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    if cam.max() > 1e-8:                      # guard against an all-zero map
        cam = cam / cam.max()
    return cam.astype(np.float32)


def attention_rollout(model, img_tensor) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020).

    For every block: average the heads, add the identity (the residual connection),
    re-normalise the rows to sum to 1. Multiply these matrices from the first block
    to the last. Row 0 (the CLS token) then says how much each of the 256 spatial
    tokens contributed to the CLS token that feeds the classifier.
    It does not depend on the target class: it shows where the model looked.
    """
    model.eval()
    x = img_tensor.to(next(model.parameters()).device)

    set_save = getattr(model, "set_save_attn", None)
    if set_save is not None:
        set_save(True)                                  # blocks only keep attention when asked
    try:
        with torch.no_grad():
            model(x)
            attns = list(model.get_attentions())
    finally:
        if set_save is not None:
            set_save(False)                             # back to normal (saves memory)

    if any(a is None for a in attns):
        raise RuntimeError("Attention maps were not stored. Check set_save_attn / get_attentions "
                           "in models/hybrid.py and the attention class in models/vit.py.")

    with torch.no_grad():
        result = None
        for attn in attns:                              # each (B, heads, N, N)
            a = attn.detach().mean(dim=1)[0]            # average heads -> (N, N)
            a = a + torch.eye(a.shape[-1], device=a.device)   # add residual path
            a = a / a.sum(dim=-1, keepdim=True)         # rows sum to 1 again
            result = a if result is None else a @ result
        cls_to_tokens = result[0, 1:]                   # (256,) skip the CLS column
        side = int(round(cls_to_tokens.numel() ** 0.5))  # 16
        cam = cls_to_tokens.reshape(1, 1, side, side)
        cam = F.interpolate(cam, size=(config.IMG_SIZE, config.IMG_SIZE),
                            mode="bilinear", align_corners=False)
    return _normalize_map(cam[0, 0].cpu().numpy())


def _stem_target_layer(model):
    """Last ReLU of the last ConvBlock in the stem (before its max-pool)."""
    last = model.stem[-1]
    if hasattr(last, "relu2"):                          # hand-written ConvBlock with named layers
        return last.relu2
    return last.block[5]                                # models/cnn.py ConvBlock: Sequential, ReLU at 5


def gradcam_stem(model, img_tensor, target_class=None) -> np.ndarray:
    """Grad-CAM on the last ReLU of the CNN stem (32x32 maps, 128 channels)."""
    engine = GradCAM(model, target_layer=_stem_target_layer(model))
    try:
        heatmap, _, _ = engine(img_tensor, target_class)
    finally:
        engine.remove()
    return heatmap


def explain(model, img_tensor, target_class=None, method: str = "rollout") -> np.ndarray:
    """Contract function. img_tensor: normalized (1,3,128,128). Returns (128,128) float32 in [0,1]."""
    if method == "rollout":
        return attention_rollout(model, img_tensor)
    if method == "gradcam":
        return gradcam_stem(model, img_tensor, target_class)
    raise ValueError(f"Unknown method: {method} (use 'rollout' or 'gradcam')")


if __name__ == "__main__":
    # Self-test with random weights: python -m xai.hybrid_explain
    from models.hybrid import BrainHybrid

    m = BrainHybrid(num_classes=config.NUM_CLASSES).eval()
    x = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    for meth in ("rollout", "gradcam"):
        h = explain(m, x, method=meth)
        ok = (h.shape == (128, 128) and h.dtype == np.float32
              and 0.0 <= float(h.min()) and float(h.max()) <= 1.0)
        print(f"{meth:8s} shape={h.shape} dtype={h.dtype} min={h.min():.3f} max={h.max():.3f} OK={ok}")