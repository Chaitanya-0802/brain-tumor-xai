"""Run from repo root:  python -m xai.sanity_check_vit --ckpt vit_v1.pt"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from data.dataset import get_transforms, seed_everything
from models.vit import BrainViT
from xai.attn_rollout import explain
from xai.gradcam import denormalize, overlay_heatmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="vit_v1.pt", help="file name inside CKPT_DIR")
    ap.add_argument("--per_class", type=int, default=2)
    args = ap.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BrainViT(num_classes=config.NUM_CLASSES).to(device)
    ck = torch.load(config.CKPT_DIR / args.ckpt, map_location=device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    random_model = BrainViT(num_classes=config.NUM_CLASSES).to(device).eval()  # untrained

    tfm = get_transforms(train=False)
    rows = []
    for cls in config.CLASSES:
        files = sorted((config.DATA_DIR / "Testing" / cls).glob("*.jpg"))
        for p in random.sample(files, args.per_class):
            rows.append((cls, p))

    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)))
    corrs = []
    for r, (true_cls, path) in enumerate(rows):
        x = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        img = denormalize(x)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        hm = explain(model, x)
        hm_rand = explain(random_model, x)
        corrs.append(np.corrcoef(hm.ravel(), hm_rand.ravel())[0, 1])

        axes[r, 0].imshow(img); axes[r, 0].set_title(f"True: {true_cls}")
        axes[r, 1].imshow(hm, cmap="jet"); axes[r, 1].set_title("Rollout heatmap")
        axes[r, 2].imshow(overlay_heatmap(img, hm))
        axes[r, 2].set_title(f"Pred: {config.CLASSES[pred]} ({probs[pred]:.0%})")
        axes[r, 3].imshow(overlay_heatmap(img, hm_rand)); axes[r, 3].set_title("Random weights")
        for a in axes[r]:
            a.axis("off")

    plt.tight_layout()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "rollout_sanity.png"
    plt.savefig(out, dpi=120)
    print("Saved:", out)
    print(f"Mean correlation trained vs random-weights heatmaps: {np.nanmean(corrs):.3f} "
          "(should be low, ideally below ~0.5)")


if __name__ == "__main__":
    main()
