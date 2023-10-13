"""
This module defines a CNN model architecture for image classification.
"""

from collections import OrderedDict

import torch
from torch import nn


class CNNModel(nn.Module):
    """
    CNNModel class defines a simple convolutional neural network architecture with torch.

    The architecture consists of alternating convolutional and dropout layers followed by ReLU activation functions,
    and a fully connected layer for final classification.

    """

    def __init__(self, num_classes: int = 2):
        """
        :param num_classes: Number of classes for classification.
        """
        super().__init__()
        self.model = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(3, 16, 3)),
                ("drop1", nn.Dropout(0.4)),
                ("relu1", nn.ReLU()),
                ("conv2", nn.Conv2d(16, 32, 3)),
                ("drop2", nn.Dropout(0.5)),
                ("relu2", nn.ReLU()),
                ("flatten", nn.Flatten()),
                ("fc", nn.Linear(1548800, num_classes))
            ])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model.

        :param x: Input tensor.
        :returns: Output tensor.
        """
        return self.model(x)