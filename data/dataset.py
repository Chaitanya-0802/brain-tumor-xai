import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def seed_everything(seed: int = config.SEED):
    """Make runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(train: bool):
    """Training gets mild augmentation. Val/test/GUI get only resize + normalize."""
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def build_samples(split_dir: Path):
    """Return a list of (image_path, label_index) for one split folder."""
    samples = []
    for label, cls in enumerate(config.CLASSES):
        cls_dir = split_dir / cls
        assert cls_dir.exists(), f"Missing folder: {cls_dir}. Check config.DATA_DIR and CLASSES."
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), label))
    return samples


class MRIDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")   # grayscale MRI -> 3 channels
        return self.transform(img), label



def get_dataloaders(batch_size=64, num_workers=2, val_frac=0.15, augment=True):
    train_all = build_samples(config.DATA_DIR / "Training")
    test_samples = build_samples(config.DATA_DIR / "Testing")

    labels = [s[1] for s in train_all]
    train_samples, val_samples = train_test_split(
        train_all, test_size=val_frac, stratify=labels, random_state=config.SEED
    )

    train_ds = MRIDataset(train_samples, get_transforms(train=augment))
    val_ds = MRIDataset(val_samples, get_transforms(train=False))
    test_ds = MRIDataset(test_samples, get_transforms(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick self-test: run with `python -m data.dataset` from the repo root
    seed_everything()
    tr, va, te = get_dataloaders()
    print("train/val/test images:", len(tr.dataset), len(va.dataset), len(te.dataset))
    x, y = next(iter(tr))
    print("batch:", x.shape, y.shape, "labels:", y[:10].tolist())