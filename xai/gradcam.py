"""Grad-CAM implemented from scratch with hooks (no external XAI library).

Team contract:
    explain(model, img_tensor, target_class=None) -> np.ndarray (128, 128), values in [0, 1]
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config


def get_target_layer(model):
    """Last ReLU inside the final ConvBlock of BrainCNN (output 256 x 16 x 16).
    Hooking before the max-pool gives a finer heatmap than the 8x8 block output."""
    return model.features[-1].block[5]


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else get_target_layer(model)
        self.activations = None   # feature maps A_k, shape (B, K, h, w)
        self.gradients = None     # d(score)/d(A_k), same shape
        # Forward hook: runs when the layer produces its output
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        # Register a hook on the output tensor itself; it fires during backward
        # and hands us the gradient flowing into this layer's output.
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def remove(self):
        """Detach the hook when you are done (avoids duplicate hooks)."""
        self._handle.remove()

    def __call__(self, x, target_class=None):
        """x: (1, 3, H, W) normalized tensor. Returns (heatmap, predicted_class, probs)."""
        self.model.eval()                      # eval mode (BatchNorm/Dropout fixed) ...
        x = x.to(next(self.model.parameters()).device)
        with torch.enable_grad():              # ... but gradients must be ON
            self.model.zero_grad()
            logits = self.model(x)             # forward pass -> hook stores activations
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = int(logits.argmax(1).item())
            cls = pred if target_class is None else int(target_class)
            logits[0, cls].backward()          # backward pass -> hook stores gradients

        # alpha_k: average each channel's gradient over its spatial positions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # (1, K, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)                                            # keep positive evidence
        cam = F.interpolate(cam, size=(config.IMG_SIZE, config.IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 1e-8:                   # guard against an all-zero map
            cam = cam / cam.max()
        return cam.astype(np.float32), pred, probs


def explain(model, img_tensor, target_class=None):
    """Contract function used by the GUI and evaluation code."""
    cam_engine = GradCAM(model)
    try:
        heatmap, _, _ = cam_engine(img_tensor, target_class)
    finally:
        cam_engine.remove()
    return heatmap


# ---------- visualization helpers ----------
def denormalize(img_tensor):
    """(1,3,H,W) or (3,H,W) normalized tensor -> (H,W,3) uint8 RGB image."""
    t = img_tensor.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    mean = torch.tensor(config.MEAN).view(3, 1, 1)
    std = torch.tensor(config.STD).view(3, 1, 1)
    t = (t * std + mean).clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay_heatmap(img_rgb, heatmap, alpha=0.4):
    """Blend a [0,1] heatmap (jet colormap) over an RGB uint8 image."""
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)       # OpenCV is BGR
    out = (1 - alpha) * img_rgb + alpha * colored
    return out.clip(0, 255).astype(np.uint8)