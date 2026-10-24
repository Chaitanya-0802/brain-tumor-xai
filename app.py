"""Gradio GUI: CNN / ViT / Hybrid / Compare All, with explanations (Member 3).

Usage (from the repo root):
    python app.py --share            # Colab: prints a public link
    python app.py --dummy            # test the GUI before training: random weights for missing checkpoints
    python app.py --cpu              # CPU fallback
    python app.py --tag v1 --hybrid_tag test   # choose checkpoint tags

EDUCATIONAL PROJECT. NOT A DIAGNOSTIC TOOL.
"""
import argparse
import threading

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from data.dataset import get_transforms
from train import get_model
from xai.gradcam import denormalize, explain as gradcam_explain, overlay_heatmap
from xai.hybrid_explain import explain as hybrid_explain

PRETTY = {"cnn": "CNN", "vit": "ViT", "hybrid": "Hybrid"}
KEY = {v: k for k, v in PRETTY.items()}
METHOD_NAME = {"cnn": "Grad-CAM", "vit": "attention rollout", "hybrid": "attention rollout"}
HYBRID_METHODS = {"Attention rollout (transformer)": "rollout",
                  "Grad-CAM (CNN stem)": "gradcam"}

DISCLAIMER = (
    "### ⚠️ Educational project: NOT a diagnostic tool\n"
    "This demo was built for a university course (CSC 507). It must not be used for medical "
    "decisions. Predictions can be wrong, and the heatmaps show what the model used, "
    "not where a tumor really is."
)

MODELS = {}          # name -> model in eval mode
UNTRAINED = set()    # names running with random weights (--dummy)
DEVICE = torch.device("cpu")
LOCK = threading.Lock()          # hooks / stored attention are per-model state: one request at a time
TF = get_transforms(train=False) # exactly the preprocessing used for validation and test


# ---------------------------------------------------------------- model loading
def load_models(tags, dummy):
    for name in PRETTY:
        path = config.CKPT_DIR / f"{name}_{tags[name]}.pt"
        try:
            model = get_model(name)
        except Exception as e:                    # e.g. models/vit.py still empty
            print(f"[skip] {name}: cannot build model ({type(e).__name__}: {e})")
            continue
        if path.exists():
            ck = torch.load(path, map_location=DEVICE)
            try:
                model.load_state_dict(ck["state_dict"])
            except RuntimeError:
                print(f"[skip] {name}: {path.name} does not match the current model class (old checkpoint?)")
                continue
        elif dummy:
            UNTRAINED.add(name)
            print(f"[dummy] {name}: no checkpoint, using UNTRAINED random weights")
        else:
            print(f"[skip] {name}: checkpoint not found ({path.name})")
            continue
        MODELS[name] = model.to(DEVICE).eval()
    if not MODELS:
        raise SystemExit("No model could be loaded. Check CKPT_DIR / --tag, or use --dummy.")
    print("Loaded models:", list(MODELS))


def get_explainer(name, hybrid_method):
    if name == "cnn":
        return lambda m, x: gradcam_explain(m, x)
    if name == "hybrid":
        return lambda m, x: hybrid_explain(m, x, method=hybrid_method)
    if name == "vit":
        try:
            from xai.attn_rollout import explain as vit_explain   # Member 2's file
        except Exception:
            return None
        return vit_explain
    return None


# ---------------------------------------------------------------- prediction
def run_one(name, x, hybrid_method):
    """Returns probs (4,), heatmap (128,128) or None, and a note."""
    model = MODELS[name]
    with torch.no_grad():                         # plain prediction needs no gradients
        probs = torch.softmax(model(x.to(DEVICE)), dim=1)[0].cpu().numpy()
    fn = get_explainer(name, hybrid_method)
    if fn is None:
        return probs, None, "no explanation function available yet"
    try:
        return probs, fn(model, x.to(DEVICE)), ""   # NOT inside no_grad: Grad-CAM needs gradients
    except Exception as e:
        return probs, None, f"explanation failed: {type(e).__name__}: {e}"


def caption_for(name, pred_label, conf):
    tag = " [UNTRAINED]" if name in UNTRAINED else ""
    return f"{PRETTY[name]}{tag} · {METHOD_NAME[name]} · {pred_label} {conf * 100:.1f}%"


def blend(state, alpha):
    """Rebuild the gallery from cached heatmaps (so the opacity slider needs no new inference)."""
    if not state:
        return []
    items = [(state["img"], "Input (resized to 128x128)")]
    for caption, heat in state["maps"]:
        items.append((overlay_heatmap(state["img"], heat, alpha=alpha), caption))
    return items


def predict(pil_img, choice, alpha, hybrid_label):
    if pil_img is None:
        raise gr.Error("Please upload or pick an image first.")
    names = list(MODELS) if choice == "Compare All" else [KEY[choice]]
    hybrid_method = HYBRID_METHODS[hybrid_label]
    x = TF(pil_img.convert("RGB")).unsqueeze(0)    # same transform as training/test
    img_rgb = denormalize(x)

    lines = ["| Model | Prediction | Confidence | Heatmap |", "|---|---|---|---|"]
    maps, all_probs = [], []
    with LOCK:
        for n in names:
            probs, heat, note = run_one(n, x, hybrid_method)
            all_probs.append(probs)
            k = int(probs.argmax())
            lines.append(f"| {PRETTY[n]}{' (untrained)' if n in UNTRAINED else ''} | "
                         f"{config.CLASSES[k]} | {probs[k] * 100:.1f}% | "
                         f"{METHOD_NAME[n] if heat is not None else note} |")
            if heat is not None:
                maps.append((caption_for(n, config.CLASSES[k], probs[k]), heat))

    mean_probs = np.mean(all_probs, axis=0)
    label = {c: float(p) for c, p in zip(config.CLASSES, mean_probs)}
    summary = "\n".join(lines)
    if len(names) > 1:
        summary += "\n\nThe label above is the average probability over all models."
    state = {"img": img_rgb, "maps": maps}
    return label, blend(state, alpha), summary, state


def reblend(state, alpha):
    return blend(state, alpha)


# ---------------------------------------------------------------- comparison tab
def read_png(path):
    return np.array(Image.open(path).convert("RGB"))


def load_comparison():
    r = config.RESULTS_DIR
    csv = r / "eval_comparison.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df["model"] = df["model"].map(lambda m: PRETTY.get(m, m))
        df["test_acc"] = (df["test_acc"] * 100).round(2)
        df["macro_f1"] = df["macro_f1"].round(4)
        df["params"] = (df["params"] / 1e6).round(2)
        df["train_time_sec"] = (df["train_time_sec"] / 60).round(1)
        df = df.rename(columns={"model": "Model", "tag": "Tag", "test_acc": "Test acc (%)",
                                "macro_f1": "Macro-F1", "params": "Params (M)",
                                "train_time_sec": "Train time (min)",
                                "infer_ms_gpu": "GPU ms/img", "infer_ms_cpu": "CPU ms/img"})
        df = df.round(2)
        status = "Results from `evaluate.py` on the held-out TEST set."
    else:
        df = pd.DataFrame({"info": ["Run: python evaluate.py --tag v1"]})
        status = "No evaluation results found yet."
    bars = r / "eval_comparison_bars.png"
    bars_img = read_png(bars) if bars.exists() else None
    cms = [(read_png(p), p.stem.replace("eval_cm_", "")) for p in sorted(r.glob("eval_cm_*.png"))]
    return df, bars_img, cms, status


HOW_IT_WORKS = """
## Grad-CAM (CNN, and the CNN stem of the hybrid)
1. Run the image through the network and pick a class score (normally the predicted class).
2. Go backwards from that score to the last convolutional feature maps. The gradient tells how much
   each feature map channel increased the score.
3. Average the gradient of each channel to get one weight per channel.
4. Take the weighted sum of the maps, keep only positive values (ReLU), upsample to the image size
   and scale to 0..1. Red = regions that pushed the score up most.

## Attention rollout (ViT, and the transformer part of the hybrid)
A transformer layer lets every token look at every other token. For the CLS token, which feeds the
classifier, the attention weights say which image tokens it reads.
1. Average the attention heads of a layer.
2. Add the identity matrix (the residual connection) and re-normalise each row.
3. Multiply these matrices layer by layer, so influence that travels through several layers is included.
4. Take the CLS row, reshape the token scores to the token grid (16x16 for the hybrid), upsample, scale to 0..1.
Rollout does not depend on a chosen class: it shows where the model looked, not why it chose a class.

## The three models (all written from scratch)
* **CNN**: stacked convolutions learn local patterns, global average pooling, linear head.
* **ViT**: the image is cut into patches that become tokens; a transformer encoder with a CLS token classifies.
* **Hybrid**: a small CNN stem turns the image into a 16x16 feature map (256 tokens), then a transformer encoder
  with a CLS token works on those tokens. Local features first, global relations second.

## Limits
* Trained on one public Kaggle dataset (128x128 images, 4 classes). It may not work on other scanners or views.
* A heatmap is an explanation of the model, not proof of a tumor location.
* **Educational project, not a diagnostic tool.**
"""


# ---------------------------------------------------------------- GUI
def example_images():
    out = []
    for c in config.CLASSES:
        d = config.DATA_DIR / "Testing" / c
        if d.exists():
            files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")) + sorted(d.glob("*.jpeg"))
            if files:
                out.append([str(files[0])])
    return out


def build_demo():
    choices = [PRETTY[n] for n in MODELS] + (["Compare All"] if len(MODELS) > 1 else [])
    with gr.Blocks(title="Brain Tumor MRI: CNN vs ViT vs Hybrid") as demo:
        gr.Markdown("# Brain Tumor MRI classification: CNN vs Vision Transformer vs Hybrid\n"
                    "Explainable AI demo (CSC 507, IIT Roorkee)")
        gr.Markdown(DISCLAIMER)
        if UNTRAINED:
            gr.Markdown(f"**Test mode:** {', '.join(PRETTY[n] for n in sorted(UNTRAINED))} "
                        "running with untrained random weights, so its predictions are meaningless.")

        with gr.Tab("Predict"):
            state = gr.State(None)
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="pil", label="MRI image (upload)")
                    model_dd = gr.Dropdown(choices, value=choices[0], label="Model")
                    hybrid_rd = gr.Radio(list(HYBRID_METHODS), value=list(HYBRID_METHODS)[0],
                                         label="Hybrid heatmap method")
                    alpha = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Heatmap opacity")
                    btn = gr.Button("Predict", variant="primary")
                    ex = example_images()
                    if ex:
                        gr.Examples(examples=ex, inputs=[img_in],
                                    label="Example test images (one per class)")
                with gr.Column():
                    label_out = gr.Label(num_top_classes=config.NUM_CLASSES, label="Prediction")
                    gallery = gr.Gallery(label="Input and explanation heatmaps", columns=2,
                                         object_fit="contain", height="auto")
                    summary = gr.Markdown()
            gr.Markdown("*Educational project, not a diagnostic tool.*")
            btn.click(predict, [img_in, model_dd, alpha, hybrid_rd],
                      [label_out, gallery, summary, state])
            alpha.change(reblend, [state, alpha], gallery)

        with gr.Tab("Model comparison"):
            df0, bars0, cms0, status0 = load_comparison()
            status = gr.Markdown(status0)
            table = gr.Dataframe(value=df0, interactive=False, label="Test-set results")
            bars = gr.Image(value=bars0, label="Comparison chart", interactive=False)
            cm_gal = gr.Gallery(value=cms0, label="Confusion matrices", columns=3,
                                object_fit="contain", height="auto")
            refresh = gr.Button("Reload results")
            refresh.click(load_comparison, None, [table, bars, cm_gal, status])

        with gr.Tab("How it works"):
            gr.Markdown(HOW_IT_WORKS)
        gr.Markdown("---\n" + DISCLAIMER)
    return demo


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    for n in PRETTY:
        ap.add_argument(f"--{n}_tag", default=None)
    ap.add_argument("--share", action="store_true", help="public link (use in Colab)")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--dummy", action="store_true", help="use random weights if a checkpoint is missing")
    args = ap.parse_args()

    DEVICE = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", DEVICE)
    tags = {n: getattr(args, f"{n}_tag") or args.tag for n in PRETTY}
    load_models(tags, args.dummy)

    demo = build_demo()
    allowed = [str(p) for p in (config.DATA_DIR, config.RESULTS_DIR) if p.exists()]
    demo.queue().launch(share=args.share, allowed_paths=allowed)


if __name__ == "__main__":
    main()