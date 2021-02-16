from typing import Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    r"""
    A simple module whose forward method is simply a lambda function, as specified in the constructor.

    Inspired by the code of Y. Idelbayev
    at https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py#L45
    """
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3,
        stride=stride, padding=1,  # Don't forget the padding!
        bias=False  # because it is usually followed by a batch norm
    )


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=1,
        stride=stride,
        bias=False  # because it is usually followed by a batch norm
    )

