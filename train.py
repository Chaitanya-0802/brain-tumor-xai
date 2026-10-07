import argparse
import json
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

import config
from data.dataset import get_dataloaders, seed_everything


def get_model(name: str) -> nn.Module:
    """Class names are part of the team contract."""
    if name == "cnn":
        from models.cnn import BrainCNN
        return BrainCNN(num_classes=config.NUM_CLASSES)
    if name == "vit":
        from models.vit import BrainViT
        return BrainViT(num_classes=config.NUM_CLASSES)
    if name == "hybrid":
        from models.hybrid import BrainHybrid
        return BrainHybrid(num_classes=config.NUM_CLASSES)
    raise ValueError(f"Unknown model: {name}")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    preds_all, labels_all = [], []

    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds_all += out.argmax(1).cpu().tolist()
            labels_all += y.cpu().tolist()

    acc = sum(p == l for p, l in zip(preds_all, labels_all)) / n
    f1 = f1_score(labels_all, preds_all, average="macro")
    return total_loss / n, acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cnn", "vit", "hybrid"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tag", default="v1", help="checkpoint version suffix, e.g. v1, v2")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    ap.add_argument("--no_aug", action="store_true", help="disable training augmentation (ablation)")
    args = ap.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    lr = args.lr or (1e-3 if args.model == "cnn" else 3e-4)
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{args.model}_{args.tag}"
    best_path = config.CKPT_DIR / f"{name}.pt"          # best weights (used by GUI)
    last_path = config.CKPT_DIR / f"{name}_last.pt"     # for resuming
    hist_path = config.RESULTS_DIR / f"{name}_history.json"
    metrics_path = config.RESULTS_DIR / f"{name}_metrics.json"

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size,  augment=not args.no_aug)
    model = get_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, start_epoch, best_val_loss, bad_epochs = [], 0, float("inf"), 0

    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        history, start_epoch = ck["history"], ck["epoch"] + 1
        best_val_loss, bad_epochs = ck["best_val_loss"], ck["bad_epochs"]
        print(f"Resumed from epoch {start_epoch}")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append(dict(epoch=epoch + 1, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1))
        print(f"Epoch {epoch+1:03d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}")

        if va_loss < best_val_loss:
            best_val_loss, bad_epochs = va_loss, 0
            torch.save({"state_dict": model.state_dict(), "classes": config.CLASSES,
                        "img_size": config.IMG_SIZE, "epoch": epoch + 1,
                        "val_loss": va_loss, "val_acc": va_acc}, best_path)
            print("  saved best ->", best_path.name)
        else:
            bad_epochs += 1

        # every epoch: save resumable state + history to Drive
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "history": history, "best_val_loss": best_val_loss,
                    "bad_epochs": bad_epochs}, last_path)
        hist_path.write_text(json.dumps(history, indent=2))

        if bad_epochs >= args.patience:
            print(f"Early stopping: no val-loss improvement for {args.patience} epochs.")
            break

    train_time = time.time() - t0

    # Final evaluation on the untouched test set, using the BEST weights
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck["state_dict"])
    te_loss, te_acc, te_f1 = run_epoch(model, test_loader, criterion, device)
    print(f"\nTEST | loss {te_loss:.4f} acc {te_acc:.4f} macro-F1 {te_f1:.4f}")

    metrics = dict(model=args.model, tag=args.tag, params=n_params,
                   best_epoch=ck["epoch"], test_acc=te_acc, test_f1=te_f1,
                   train_time_sec=round(train_time, 1))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("Saved metrics ->", metrics_path)


if __name__ == "__main__":
    main()