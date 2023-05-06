from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from .resnet import ResNet

__all__ = [
    "resnet10",
    "resnet12",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "default_matching_networks_support_encoder",
    "default_matching_networks_query_encoder",
    "default_relation_module",
]


def resnet10(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet12(**kwargs):
    return ResNet(BasicBlock, [1, 1, 2, 1], planes=[64, 160, 320, 640], **kwargs)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def default_matching_networks_support_encoder(feature_dimension: int) -> nn.Module:
    return nn.LSTM(
        input_size=feature_dimension,
        hidden_size=feature_dimension,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
    )


def default_matching_networks_query_encoder(feature_dimension: int) -> nn.Module:
    return nn.LSTMCell(feature_dimension * 2, feature_dimension)


def default_relation_module(feature_dimension: int, inner_channels: int = 8):
    
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(
                feature_dimension * 2,
                feature_dimension,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((5, 5)),
        ),
        nn.Sequential(
            nn.Conv2d(
                feature_dimension,
                feature_dimension,
                kernel_size=3,
                padding=0,
            ),
            nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
        ),
        nn.Flatten(),
        nn.Linear(feature_dimension, inner_channels),
        nn.ReLU(),
        nn.Linear(inner_channels, 1),
        nn.Sigmoid(),
    )