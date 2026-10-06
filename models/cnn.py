import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class BrainCNN(nn.Module):
    """Input (B,3,128,128) -> logits (B,4)."""
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),     # 128 -> 64
            ConvBlock(32, 64),    # 64 -> 32
            ConvBlock(64, 128),   # 32 -> 16
            ConvBlock(128, 256),  # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))