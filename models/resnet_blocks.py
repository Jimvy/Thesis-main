from typing import List, Sequence

import torch.nn as nn

from .utils import conv1x1, conv3x3


class ResidualBlock(nn.Module):
    r"""
    Base class for all residual blocks.

    A residual block is defined here as the stack of layers that is present in the non-shortcut path of a residual unit.
    For a ResNet, we usually have

    .. math:: x_{l+1} = g(x_l + f(x_l))

    where :math:`g` is the activation function, :math:`x_l` is the input of the residual unit,
    :math:`x_{l+1}` is the output. Then, :math:`f` is what we call here the residual block.

    The main characteristic of a residual block is that its input and output tensors for each layer (except the first)
    have the same shape, in order to perform the pointwise addition.

    It is possible for a residual block to perform a downsampling: e.g. perform a convolution with a stride.
    """

    def __init__(self, in_channels, channels, stride=1, **kwargs):
        super(ResidualBlock, self).__init__()
        self._in_channels = in_channels
        self._channels = channels
        self._stride = stride
        self.layers = nn.ModuleList()

    def add_layer(self, new_layer: nn.Module):
        self.layers.append(new_layer)

    def add_layers(self, new_layers: Sequence[nn.Module]):
        self.layers += new_layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BilayerBlock(ResidualBlock):
    r"""
    Basic block for a ResNet, consisting of two identical weight layers.

    Such a basic block is used in ResNet-18 and ResNet-34 on ImageNet, and in ResNet-20, -32, -44 and -56 on CIFAR.

    When performing a downsampling, it performs a strided convolution in the first layer.
    """

    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, stride=1):
        super(BilayerBlock, self).__init__(in_channels, channels, stride)
        self.add_layers([
            conv3x3(in_channels, channels, stride=stride),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            conv3x3(channels, channels),
            norm_layer(channels),
        ])


class PreActBilayerBlock(ResidualBlock):
    r"""
    Basic block for a ResNet, consisting of two identical (convolutional) weight layers, using pre-activation.
    """


class BottleneckBlock(ResidualBlock):
    r"""
    Bottleneck block, as defined in the original paper on ResNet.
    A bottleneck block consists of a 1x1 convolution with a reduced number of output channels,
    followed by a 3x3 convolution with the same number of input and output channels (thus consisting of a bottleneck),
    followed by a 1x1 convolution which restores the original number of channels at its output.

    We refer to "inner channel number" as the number of channels at the input and output of the 3x3 convolution,
    which thus constitutes a bottleneck.

    When performing a downsampling, the strided convolution is performed on the 3x3 convolution.
    """

    default_expansion = 4  # Ratio between #output channels and #inner channels

    def __init__(self, in_channels, channels, expansion=4, norm_layer=nn.BatchNorm2d, stride=1):
        r"""
        Constructor.
        :param in_channels: number of input channels to this block.
        :param channels: number of output channels of this module;
                         this is the number of channels during the element-wise operations
        :param expansion: ratio of number of output channels of this block to number of inner channels of this block.
        """
        super(BottleneckBlock, self).__init__(in_channels, channels, stride)
        inner_channels = channels // expansion
        self.add_layers([
            conv1x1(in_channels, inner_channels),
            norm_layer(inner_channels),
            nn.ReLU(inplace=True),
            conv3x3(inner_channels, inner_channels, stride=stride),
            norm_layer(inner_channels),
            nn.ReLU(inplace=True),
            conv1x1(inner_channels, channels),
            norm_layer(channels)
        ])


class InvertedBottleneckBlock(ResidualBlock):
    r"""
    Inverted residual bottleneck block, as defined in MobileNetV2.
    TODO
    """
