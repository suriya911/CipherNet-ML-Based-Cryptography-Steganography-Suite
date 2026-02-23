from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return feat, self.pool(feat)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class StegoUNet(nn.Module):
    """U-Net-like encoder with dual decoders for stego residual and secret recovery."""

    def __init__(self, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.residual_scale = residual_scale

        # Cover encoder (3x256x256 -> bottleneck at 16x16)
        self.cover_down1 = DownBlock(3, 64)    # 256 -> 128
        self.cover_down2 = DownBlock(64, 128)  # 128 -> 64
        self.cover_down3 = DownBlock(128, 256) # 64 -> 32
        self.cover_down4 = DownBlock(256, 512) # 32 -> 16

        # Secret embedding path (3x64x64 -> 512x16x16)
        self.secret_encoder = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(2),  # 64 -> 32
            ConvBlock(64, 128),
            nn.MaxPool2d(2),  # 32 -> 16
            ConvBlock(128, 256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Fused bottleneck
        self.bottleneck = ConvBlock(1024, 512)

        # Stego residual decoder to 256x256x3
        self.stego_up4 = UpBlock(512, 512, 256)  # 16 -> 32
        self.stego_up3 = UpBlock(256, 256, 128)  # 32 -> 64
        self.stego_up2 = UpBlock(128, 128, 64)   # 64 -> 128
        self.stego_up1 = UpBlock(64, 64, 32)     # 128 -> 256
        self.stego_head = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh(),
        )

        # Secret decoder to 64x64x3
        self.secret_up2 = UpBlock(512, 512, 256)  # 16 -> 32 (skip c4)
        self.secret_up1 = UpBlock(256, 256, 128)  # 32 -> 64 (skip c3)
        self.secret_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, cover: torch.Tensor, secret: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode cover with skip connections
        c1, p1 = self.cover_down1(cover)
        c2, p2 = self.cover_down2(p1)
        c3, p3 = self.cover_down3(p2)
        c4, p4 = self.cover_down4(p3)

        # Encode secret (upsample secret to expected input size if needed)
        if secret.shape[-2:] != (64, 64):
            secret = F.interpolate(secret, size=(64, 64), mode="bilinear", align_corners=False)
        s = self.secret_encoder(secret)

        fused = self.bottleneck(torch.cat([p4, s], dim=1))

        # Stego path
        x = self.stego_up4(fused, c4)
        x = self.stego_up3(x, c3)
        x = self.stego_up2(x, c2)
        x = self.stego_up1(x, c1)
        residual = self.stego_head(x) * self.residual_scale
        stego = torch.clamp(cover + residual, 0.0, 1.0)

        # Secret recovery path
        z = self.secret_up2(fused, c4)
        z = self.secret_up1(z, c3)
        recovered_secret = self.secret_head(z)

        return stego, recovered_secret, residual
