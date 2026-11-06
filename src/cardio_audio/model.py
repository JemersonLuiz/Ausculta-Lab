"""CNN compacta; pesos somente são gerados pelo treinamento documentado."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in (16, 32, 64):
            layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
                                        nn.Dropout(0.3), nn.Linear(64 * 4 * 8, 64),
                                        nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))


class BandCNN(nn.Module):
    """Extrai padrões locais e resume energia típica e picos por faixa."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=(2, 4), padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2 * 64 * 4 * 2, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 4),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.features((values - 0.5) * 3)
        average = F.adaptive_avg_pool2d(features, (4, 2))
        peaks = F.adaptive_max_pool2d(features, (4, 2))
        return self.classifier(torch.cat((average, peaks), dim=1).flatten(1))


def build_model(architecture: str = "small_cnn_v1") -> nn.Module:
    if architecture == "small_cnn_v1":
        return SmallCNN()
    if architecture == "band_cnn_v2":
        return BandCNN()
    raise ValueError(f"Arquitetura incompatível: {architecture}")

