from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.depthwise(x))
        x = self.activation(self.pointwise(x))
        return x


class CPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.expand = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.expand(x))
        x = self.pixel_shuffle(x)
        return x


class SelectiveFeatureAttention(nn.Module):
    def __init__(self, channels: int = 256, reduction: int = 16, min_channels: int = 32) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, min_channels)
        self.compress = nn.Linear(channels, reduced_channels)
        self.expand = nn.Linear(reduced_channels, channels * 2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, higher_feature: Tensor, current_feature: Tensor) -> Tensor:
        # ---------------------------------------------------------------------
        # Align the higher-semantic map to the larger spatial resolution
        # ---------------------------------------------------------------------
        if higher_feature.shape[-2:] != current_feature.shape[-2:]:
            higher_feature = F.interpolate(higher_feature, size=current_feature.shape[-2:], mode="nearest")

        # ---------------------------------------------------------------------
        # Predict branch weights from the fused descriptor
        # ---------------------------------------------------------------------
        fused_feature = higher_feature + current_feature
        descriptor = F.adaptive_avg_pool2d(fused_feature, output_size=1).flatten(1)
        descriptor = self.activation(self.compress(descriptor))
        weights = self.expand(descriptor).view(descriptor.shape[0], 2, -1, 1, 1)
        weights = torch.softmax(weights, dim=1)

        # ---------------------------------------------------------------------
        # Reweight the semantic-rich and high-resolution branches
        # ---------------------------------------------------------------------
        return weights[:, 0] * higher_feature + weights[:, 1] * current_feature


class SFPSPyramid(nn.Module):
    def __init__(self, out_channels: int = 256, attention_reduction: int = 16) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.c5_projection = DepthwiseSeparableConv(2048, out_channels)
        self.c2_projection = DepthwiseSeparableConv(256, out_channels)

        self.c5_cp = CPBlock(2048, out_channels)
        self.c4_cp = CPBlock(1024, out_channels)
        self.c3_cp = CPBlock(512, out_channels)

        self.p3_attention = SelectiveFeatureAttention(out_channels, reduction=attention_reduction)
        self.p2_attention = SelectiveFeatureAttention(out_channels, reduction=attention_reduction)

    def forward(self, features: dict[str, Tensor]) -> OrderedDict[str, Tensor]:
        c2 = features["c2"]
        c3 = features["c3"]
        c4 = features["c4"]
        c5 = features["c5"]

        # ---------------------------------------------------------------------
        # Build the semantic-heavy feature maps first
        # ---------------------------------------------------------------------
        p5 = self.c5_projection(c5)
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2)
        p4 = self.c5_cp(c5)

        # ---------------------------------------------------------------------
        # Reconstruct the larger feature maps with cross-scale attention
        # ---------------------------------------------------------------------
        p3 = self.c4_cp(c4)
        p2 = self.c3_cp(c3)
        p1 = self.c2_projection(c2)

        p3_prime = self.p3_attention(p4, p3)
        p2_prime = self.p2_attention(p3_prime, p1 + p2)

        return OrderedDict(
            {
                "p2": p2_prime,
                "p3": p3_prime,
                "p4": p4,
                "p5": p5,
                "p6": p6,
            }
        )
