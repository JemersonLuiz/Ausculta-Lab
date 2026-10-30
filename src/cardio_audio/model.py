"""CNN compacta; pesos somente são gerados pelo treinamento documentado."""
from __future__ import annotations

import torch
from torch import nn


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

