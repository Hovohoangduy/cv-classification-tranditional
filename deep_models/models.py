"""CIFAR-sized AlexNet, VGG11, ResNet18 and a small Vision Transformer.

Only PyTorch tensor operations and building blocks are used; no model zoo,
pretrained weights, or torchvision architecture implementations are imported.
"""

from __future__ import annotations

import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2), nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 3, padding=1), nn.GroupNorm(8, 192),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1), nn.GroupNorm(8, 384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class VGG11(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # VGG11's eight 3x3 convolutions and five pooling stages, with
        # narrower channels to suit 32x32 images and CPU training.
        layout = [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"]
        layers: list[nn.Module] = []
        channels = 3
        for item in layout:
            if item == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.extend((nn.Conv2d(channels, item, 3, padding=1, bias=False),
                               nn.BatchNorm2d(item), nn.ReLU(inplace=True)))
                channels = item
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == channels else
                         nn.Sequential(nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                                       nn.BatchNorm2d(channels)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)) + residual)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # CIFAR stem: no 7x7 stride-2 convolution or initial max pooling.
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            self._stage(32, 32, 1), self._stage(32, 64, 2),
            self._stage(64, 128, 2), self._stage(128, 256, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def _stage(in_channels: int, channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(BasicBlock(in_channels, channels, stride), BasicBlock(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.pool(self.layers(self.stem(x))), 1))


class TransformerBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        if dimension % heads:
            raise ValueError("embedding dimension must be divisible by heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.scale = self.head_dimension ** -0.5
        self.norm1 = nn.LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.projection = nn.Linear(dimension, dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension * mlp_ratio), nn.GELU(),
                                 nn.Linear(dimension * mlp_ratio, dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        batch, tokens, dimension = normalized.shape
        qkv = self.qkv(normalized).reshape(batch, tokens, 3, self.heads, self.head_dimension)
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (queries @ keys.transpose(-2, -1)) * self.scale
        weights = scores.softmax(dim=-1)
        attended = (weights @ values).transpose(1, 2).reshape(batch, tokens, dimension)
        x = x + self.projection(attended)
        return x + self.mlp(self.norm2(x))


class ViTTiny(nn.Module):
    def __init__(self, num_classes: int = 10, image_size: int = 32,
                 patch_size: int = 4, dimension: int = 192, depth: int = 6, heads: int = 3):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        patch_count = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dimension, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position = nn.Parameter(torch.zeros(1, patch_count + 1, dimension))
        self.blocks = nn.Sequential(*(TransformerBlock(dimension, heads) for _ in range(depth)))
        self.norm = nn.LayerNorm(dimension)
        self.head = nn.Linear(dimension, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        token = self.cls_token.expand(x.shape[0], -1, -1)
        encoded = self.blocks(torch.cat((token, patches), dim=1) + self.position)
        return self.head(self.norm(encoded[:, 0]))


MODELS = {"alexnet": AlexNet, "vgg11": VGG11, "resnet18": ResNet18, "vit_tiny": ViTTiny}


def create_model(name: str, num_classes: int = 10) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    return MODELS[name](num_classes=num_classes)
