"""Small convolutional network used by CropVision's EuroSAT experiments."""

from __future__ import annotations

import torch
from torch import nn


class SimpleEuroSATCNN(nn.Module):
    """A compact CNN for 64x64 RGB images.

    Three convolution/ReLU/pooling stages reduce the spatial resolution from
    64x64 to 8x8. Adaptive pooling fixes the classification-head input at 4x4,
    making the architecture explicit and inexpensive enough for an 8 GB Mac.
    """

    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, int, int] = (32, 64, 128),
        hidden_units: int = 256,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain three positive integers")
        if num_classes <= 1 or hidden_units <= 0:
            raise ValueError("num_classes and hidden_units must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        first, second, third = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, first, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(first, second, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(second, third, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(third * 4 * 4, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits for a batch of RGB tensors."""
        return self.classifier(self.features(inputs))


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters updated by the optimizer."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
