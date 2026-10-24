"""Evaluate trained models on the TEST set and build comparison outputs (Member 3).

Usage (from the repo root):
    python evaluate.py --tag v1
    python evaluate.py --tag test            # works on smoke-test checkpoints too
Models whose checkpoint (or class file) does not exist yet are skipped.

Outputs in RESULTS_DIR (all names start with eval_ so nothing from teammates is overwritten):
    eval_comparison.csv / eval_comparison_table.png / eval_comparison_bars.png
    eval_per_class.csv, eval_report_{model}_{tag}.csv / .txt
    eval_cm_{model}_{tag}.png
"""
import argparse
import copy
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")                      # draw to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config
from data.dataset import get_dataloaders, seed_everything
from train import get_model

MODEL_NAMES = ["cnn", "vit", "hybrid"]
PRETTY = {"cnn": "CNN", "vit": "ViT", "hybrid": "Hybrid"}
COLORS = {"cnn": "#0072B2", "vit": "#E69F00", "hybrid": "#009E73"}   # colour-blind safe


def load_model(name, tag, device):
    """Return the model with best weights in eval mode, or None if it cannot be loaded."""
    path = config.CKPT_DIR / f"{name}_{tag}.pt"
    if not path.exists():
        print(f"[skip] {name}: checkpoint not found ({path.name})")
        return None
    try:
        model = get_model(name)
    except Exception as e:                                   # e.g. empty models/vit.py
        print(f"[skip] {name}: cannot build model ({type(e).__name__}: {e})")
        return None
    ck = torch.load(path, map_location=device)
    try:
        model.load_state_dict(ck["state_dict"])
    except RuntimeError as e:                    # checkpoint made with a different architecture
        print(f"[skip] {name}: {path.name} does not match the current model class (old checkpoint?)")
        print("       first line of error:", str(e).splitlines()[0])
        return None
    return model.to(device).eval()


@torch.no_grad()
def predict_all(model, loader, device):
    probs_all, labels_all = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(y.numpy())
    probs = np.concatenate(probs_all)
    return probs.argmax(1), np.concatenate(labels_all), probs


@torch.no_grad()
def time_inference(model, device, n_warmup=20, n_runs=100):
    """Average latency in ms for ONE image (batch size 1).

    Warm-up runs first (CUDA kernels get compiled/cached, caches fill). GPU work is
    asynchronous, so we call torch.cuda.synchronize() before starting and before stopping
    the clock; otherwise we would only time how fast the work was *queued*.
    """
    model = model.to(device).eval()
    x = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000.0


def plot_confusion(cm, title, path):
    n = cm.shape[0]
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(config.CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(config.CLASSES)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.1f}%", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(v, spec):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def save_table_png(df, path):
    show = pd.DataFrame({
        "Model": df["model"].map(PRETTY),
        "Test accuracy (%)": [fmt(v * 100, ".2f") for v in df["test_acc"]],
        "Macro-F1": [fmt(v, ".4f") for v in df["macro_f1"]],
        "Params (M)": [fmt(v / 1e6, ".2f") for v in df["params"]],
        "Train time (min)": [fmt(v / 60 if v == v and v is not None else None, ".1f")
                             for v in df["train_time_sec"]],
        "Inference GPU (ms/img)": [fmt(v, ".2f") for v in df["infer_ms_gpu"]],
        "Inference CPU (ms/img)": [fmt(v, ".2f") for v in df["infer_ms_cpu"]],
    })
    fig, ax = plt.subplots(figsize=(11.5, 1.3 + 0.55 * len(show)))
    ax.axis("off")
    tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E8EEF4")
    ax.set_title("Model comparison on the TEST set", fontsize=12, pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_bar_chart(df, path):
    names = [PRETTY[m] for m in df["model"]]
    cols = [COLORS[m] for m in df["model"]]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]                                     # accuracy and macro-F1 (axis starts at 80)
    w = 0.38
    b1 = ax.bar(x - w / 2, df["test_acc"] * 100, w, color=cols, label="Accuracy (%)")
    b2 = ax.bar(x + w / 2, df["macro_f1"] * 100, w, color=cols, alpha=0.55, hatch="//",
                label="Macro-F1 x100")
    ax.bar_label(b1, fmt="%.1f", fontsize=8)
    ax.bar_label(b2, fmt="%.1f", fontsize=8)
    ax.set_ylim(80, 100)
    ax.set_ylabel("%  (axis starts at 80)")
    ax.set_title("Accuracy (solid) and macro-F1 (hatched)")

    ax = axes[1]                                     # parameters
    b = ax.bar(x, df["params"] / 1e6, color=cols)
    ax.bar_label(b, fmt="%.2f", fontsize=9)
    ax.set_ylabel("Parameters (millions)")
    ax.set_title("Model size")

    ax = axes[2]                                     # inference time, GPU and CPU
    gpu = df["infer_ms_gpu"].astype(float).values
    cpu = df["infer_ms_cpu"].astype(float).values
    b1 = ax.bar(x - w / 2, np.nan_to_num(gpu), w, color=cols, label="GPU")
    b2 = ax.bar(x + w / 2, np.nan_to_num(cpu), w, color=cols, alpha=0.55, hatch="//", label="CPU")
    ax.bar_label(b1, labels=[fmt(v, ".1f") for v in gpu], fontsize=8)
    ax.bar_label(b2, labels=[fmt(v, ".1f") for v in cpu], fontsize=8)
    ax.set_ylabel("ms per image (batch size 1)")
    ax.set_title("Inference time: GPU (solid), CPU (hatched)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1", help="checkpoint tag used for all models")
    for n in MODEL_NAMES:
        ap.add_argument(f"--{n}_tag", default=None, help=f"override tag for {n}")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--runs", type=int, default=100, help="timed runs for inference time")
    args = ap.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_loader = get_dataloaders(batch_size=args.batch_size)[2]
    print("Test images:", len(test_loader.dataset))

    rows, per_class_rows = [], []
    for name in MODEL_NAMES:
        tag = getattr(args, f"{name}_tag") or args.tag
        model = load_model(name, tag, device)
        if model is None:
            continue
        print(f"\n===== {PRETTY[name]} ({name}_{tag}) =====")
        preds, labels, _ = predict_all(model, test_loader, device)
        acc = float((preds == labels).mean())
        f1 = float(f1_score(labels, preds, average="macro"))
        n_params = sum(p.numel() for p in model.parameters())

        # numbers written by train.py (training time cannot be recomputed here)
        train_time, reported_acc = None, None
        mpath = config.RESULTS_DIR / f"{name}_{tag}_metrics.json"
        if mpath.exists():
            m = json.loads(mpath.read_text())
            train_time, reported_acc = m.get("train_time_sec"), m.get("test_acc")
        else:
            print(f"[warn] {mpath.name} not found: training time will be n/a")
        if reported_acc is not None and abs(reported_acc - acc) > 0.005:
            print(f"[warn] test acc here {acc:.4f} differs from train.py's {reported_acc:.4f}")

        report_txt = classification_report(labels, preds, target_names=config.CLASSES, digits=4)
        print(report_txt)
        report = classification_report(labels, preds, target_names=config.CLASSES, digits=4,
                                       output_dict=True)
        pd.DataFrame(report).T.to_csv(config.RESULTS_DIR / f"eval_report_{name}_{tag}.csv")
        (config.RESULTS_DIR / f"eval_report_{name}_{tag}.txt").write_text(report_txt)
        for c in config.CLASSES:
            per_class_rows.append(dict(model=name, tag=tag, cls=c, precision=report[c]["precision"],
                                       recall=report[c]["recall"], f1=report[c]["f1-score"],
                                       support=int(report[c]["support"])))

        cm = confusion_matrix(labels, preds, labels=list(range(config.NUM_CLASSES)))
        plot_confusion(cm, f"{PRETTY[name]} ({tag}): confusion matrix, test set",
                       config.RESULTS_DIR / f"eval_cm_{name}_{tag}.png")

        gpu_ms = time_inference(model, device, n_runs=args.runs) if device.type == "cuda" else float("nan")
        cpu_model = copy.deepcopy(model).cpu()
        cpu_ms = time_inference(cpu_model, torch.device("cpu"), n_warmup=5, n_runs=30)
        print(f"Inference: GPU {gpu_ms:.2f} ms/img | CPU {cpu_ms:.2f} ms/img | params {n_params:,}")

        rows.append(dict(model=name, tag=tag, test_acc=acc, macro_f1=f1, params=n_params,
                         train_time_sec=train_time if train_time is not None else float("nan"),
                         infer_ms_gpu=gpu_ms, infer_ms_cpu=cpu_ms))

    if not rows:
        print("\nNo checkpoint could be evaluated. Check --tag and that files exist in", config.CKPT_DIR)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "eval_comparison.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(config.RESULTS_DIR / "eval_per_class.csv", index=False)
    save_table_png(df, config.RESULTS_DIR / "eval_comparison_table.png")
    save_bar_chart(df, config.RESULTS_DIR / "eval_comparison_bars.png")
    print("\n", df.to_string(index=False))
    print("\nSaved everything to", config.RESULTS_DIR)


if __name__ == "__main__":
    main()