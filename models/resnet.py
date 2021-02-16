"""
References:
    [1] K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition", arXiv:1512.03385
    [2] The PyTorch project, https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    [3] Y. Idelbayev, https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""

from typing import Tuple

import torch
import torch.nn as nn

from .resnet_blocks import *
from .resnet_unit import *
from .utils import LambdaLayer


class ResNet(nn.Module):
    def __init__(self, unit: Type[ResidualUnit], block: Type[ResidualBlock], stem: Type[nn.Module],
                 layer_config: List[Tuple],
                 num_classes: int = 10,
                 zero_init_residual=True):
        super(ResNet, self).__init__()

        self.layers = nn.ModuleList()
        self.stem = stem()
        self._in_channels = self.stem.inplanes  # TODO generalize

        for idx, layer_cfg in enumerate(layer_config):
            (num_channels, num_blocks) = layer_cfg
            self.layers.append(self._make_layer(unit, block, num_channels, num_blocks, stride=(1 if idx == 0 else 2)))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self._in_channels, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Init the last BN layer of the residual blocks with weight=0
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    last_bn = None
                    for mm in m.modules():
                        if isinstance(mm, nn.BatchNorm2d):
                            last_bn = mm
                    nn.init.constant_(last_bn.weight, 0)

    def _make_layer(self, unittype: Type[ResidualUnit], blocktype: Type[ResidualBlock],
                    channels: int, blocks: int, stride=1) -> nn.Sequential:
        r"""
        Creates a sequence of layers, of the same receptive field, with the specified type of residual blocks,
        repeated blocks time.

        :param unittype: type of residual unit used: Pre/Post activation, projection shortcuts,
                         how to handle changing image sizes, etc.
        :param blocktype: type of residual block used: dual layer, Bottleneck, Inverted Bottleneck etc.
        :param channels: number of channels in and at the output of the residual units.
        :param blocks: number of residual units that are stacked.
        :param stride: stride of the first residual unit, if needed.
        :return: a Sequence of residual units.
        """
        layers = []
        layers.append(unittype(blocktype, self._in_channels, channels, stride=stride))
        self._in_channels = channels
        for _ in range(1, blocks):
            layers.append(unittype(blocktype, self._in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ImageNet224ResNetStem(nn.Module):
    r"""
    Stem for the ResNet architecture on the ImageNet dataset.
    It takes tensors of shape 224x224x3 and outputs tensors of shape 56x56x64
    """
    def __init__(self):
        super(ImageNet224ResNetStem, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CifarResNetStem(nn.Module):
    r"""
    Stem for the ResNet architecture on the CIFAR-10 and CIFAR-100 datasets.
    It takes tensors of shape 32x32x3 and outputs tensors of shape 32x32x16
    """
    def __init__(self):
        super(CifarResNetStem, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


# ImageNet models
# The results were obtained using the protocols of 1512.03385, that is: lr=0.1, /= 10 when error plateaus,
# momentum 0.9, weight decay 1e-4. up to 60e4 iterations.
# Mini-batch size 256.
# 224x224 crop from an image with shorter size in [256, 480] (randomly cropped), horizontal flip.
# Per-pixel mean subtracted.
# Testing: 10-crop.


class ResNet18(ResNet):
    def __init__(self, **kwargs):
        super(ResNet18, self).__init__(
            unit=IdentityShortcutPostReLUResidualUnit, block=BilayerBlock, stem=ImageNet224ResNetStem,
            layer_config=[(64, 2), (128, 2), (256, 2), (512, 2)],
            num_classes=1000,
            **kwargs
        )


class ResNet34(ResNet):
    r"""
    Should have 24.52% top-1 error, 7.46% top-5 error.
    Option A gives 0.49% more top-1 error (0.30% more top-5).
    Option C gives 0.33% less top-1 error (0.06% less top-5).
    """
    def __init__(self, **kwargs):
        super(ResNet34, self).__init__(
            unit=IdentityShortcutPostReLUResidualUnit, block=BilayerBlock, stem=ImageNet224ResNetStem,
            layer_config=[(64, 3), (128, 4), (256, 6), (512, 3)],
            **kwargs
        )


class ResNet50(ResNet):
    r"""
    Should have 22.85% top-1 error, 6.71% top-5 error (using 10-crop testing).
    """
    def __init__(self, **kwargs):
        super(ResNet50, self).__init__(
            unit=IdentityShortcutPostReLUResidualUnit, block=BottleneckBlock, stem=ImageNet224ResNetStem,
            layer_config=[(256, 3), (512, 4), (1024, 6), (2048, 3)],
            **kwargs
        )


class ResNet101(ResNet):
    r"""
    Should have 21.75% top-1 error, 6.05% top-5 error.
    """
    def __init__(self, **kwargs):
        super(ResNet101, self).__init__(
            unit=IdentityShortcutPostReLUResidualUnit, block=BottleneckBlock, stem=ImageNet224ResNetStem,
            layer_config=[(256, 3), (512, 4), (1024, 23), (2048, 3)],
            **kwargs
        )


class ResNet152(ResNet):
    r"""
    Should have 21.43% top-1 error, 5.71% top-5 error.
    """
    def __init__(self, **kwargs):
        super(ResNet152, self).__init__(
            unit=IdentityShortcutPostReLUResidualUnit, block=BottleneckBlock, stem=ImageNet224ResNetStem,
            layer_config=[(256, 3), (512, 8), (1024, 36), (2048, 3)],
            **kwargs
        )


class ResNet200(ResNet):
    r"""
    The ResNet-200 using post-activations, of 1603.05027.
    Should have 21.8% top-1, 6.0 top-5 error, on a 320x320 test crop.
    """


class PreResNet152(ResNet):
    r"""
    The PreResNet-152 of 1603.05027.
    Should have 21.1% top-1 error, 5.5% top-5 error, on a 320x320 test crop.
    """
    def __init__(self, **kwargs):
        super(PreResNet152, self).__init__(
            unit=IdentityShortcutPostReLUResidualUnit, block=BottleneckBlock, stem=ImageNet224ResNetStem,
            layer_config=[(256, 3), (512, 8), (1024, 36), (2048, 3)],
            **kwargs
        )


class PreResNet200(ResNet):
    r"""
    The ResNet-200 using pre-activations, of 1603.05027.
    Should have 20.1% top-1 error, 4.8% top-5 error, on a 320x320 test crop.
    """


# CIFAR-10 models
# The results were obtained using the protocol of 1512.03385, that is: lr=0.1, /=10 at 32k and 48k iterations,
# training stopped at 64k (determined on a 45k/5k train/val split), using momentum 0.9, weight decay 1e-4,
# batch size 128 on two GPUs. 4 pixel padding on each size with 32x32 crop (random).


class ResNet20(ResNet):
    r"""
    Should have 0.27M parameters and reach 8.75% error.
    """
    def __init__(self, **kwargs):
        super(ResNet20, self).__init__(
            unit=IdentityShortcutDownPadPostReLUResidualUnit, block=BilayerBlock, stem=CifarResNetStem,
            layer_config=[(16, 3), (32, 3), (64, 3)],
            **kwargs
        )


class ResNet32(ResNet):
    r"""
    Should have 0.46M parameters and reach 7.51% error.
    """
    def __init__(self, **kwargs):
        super(ResNet32, self).__init__(
            unit=IdentityShortcutDownPadPostReLUResidualUnit, block=BilayerBlock, stem=CifarResNetStem,
            layer_config=[(16, 5), (32, 5), (64, 5)],
            **kwargs
        )


class ResNet44(ResNet):
    r"""
    Should have 0.66M parameters and reach 7.17% error.
    """
    def __init__(self, **kwargs):
        super(ResNet44, self).__init__(
            unit=IdentityShortcutDownPadPostReLUResidualUnit, block=BilayerBlock, stem=CifarResNetStem,
            layer_config=[(16, 7), (32, 7), (64, 7)],
            **kwargs
        )


class ResNet56(ResNet):
    r"""
    Should have 0.85M parameters and reach 6.97% error.
    """
    def __init__(self, **kwargs):
        super(ResNet56, self).__init__(
            unit=IdentityShortcutDownPadPostReLUResidualUnit, block=BilayerBlock, stem=CifarResNetStem,
            layer_config=[(16, 9), (32, 9), (64, 9)],
            **kwargs
        )


class ResNet110(ResNet):
    r"""
    Should have 1.7M parameters and reach 6.61% error (std 0.16%, best obtained was 6.43%).
    Needs a smaller learning rate (0.01) for the first few 400 iterations to start converging.
    """
    def __init__(self, **kwargs):
        super(ResNet110, self).__init__(
            unit=IdentityShortcutDownPadPostReLUResidualUnit, block=BilayerBlock, stem=CifarResNetStem,
            layer_config=[(16, 18), (32, 18), (64, 18)],
            **kwargs
        )


class ResNet1202(ResNet):
    r"""
    Should have 19.4M parameters and reach 7.93% error.
    """
    def __init__(self, **kwargs):
        super(ResNet1202, self).__init__(
            unit=IdentityShortcutDownPadPostReLUResidualUnit, block=BilayerBlock, stem=CifarResNetStem,
            layer_config=[(16, 200), (32, 200), (64, 200)],
            **kwargs
        )


# Wide ResNet:
# They are defined for ResNet-50 and ResNet-101,
# and they use twice the number of bottleneck inner channels in each residual block than in the standard ResNet.
# That is, where the last block of ResNet-50 has 2048-512-2048 channels, here it has 2048-1024-2048.
# So this is less a bottleneck...
# In PyTorch, this is implemented by doubling the property "width_per_Group" from 64 to 128.
# This directly changes the base_width and thus the inner width of Bottleneck layers.
# To implement this here, maybe use an additional parameter to change the expansion factor?

# ResNeXt:
# They are defined for ResNet-50 and ResNet-101,
# and they use grouped convolution: there are 32 groups, and 4 (for ResNeXt-50) or 8 (ResNeXt-101) channels per group.
# The number of groups is used in the 3x3 convolution of the bottleneck block to implemented grouped convolution.
# In PyTorch, this is implemented by setting the "groups" property to 32 and modifying the "width_per_group" property
# to 4 or 8. This will change the computation of the "width" in the Bottleneck blocks.

# Dilation:
# In the PyTorch models, it is possible to implement dilated convolution by asking the class to replace strides
# with dilations. The dilations are accumulated over the layers.

# Other:
# In the PyTorch models, it is possible to change the normalization layer type used.
