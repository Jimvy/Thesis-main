from typing import Dict, Type

import torch.nn as nn
import torch.nn.functional as F

from .resnet_blocks import ResidualBlock
from .utils import LambdaLayer


class ResidualUnit(nn.Module):
    r"""
    Base class for all residual units.

    A residual unit is defined here as the ensemble of layers comprising both the residual block in the residual path,
    the blocks in the shortcut path, the merge operation (typically addition), and other transformations of the signals.

    The main characteristic of a residual unit is that, if the receptive field size remains the same,
    then the input and output tensors have the same shape (same number of channels).

    It is possible for a residual unit to perform downsampling, to change the receptive field size.
    For this, both the residual block and the shortcut path need to have the same characteristics.
    """
    def __init__(self, residual_block_type: Type[ResidualBlock], in_channels, channels, stride=1,
                 resblock_extra_kwargs: Dict = None):
        super(ResidualUnit, self).__init__()
        if not resblock_extra_kwargs:
            resblock_extra_kwargs = {}
        self._in_channels = in_channels
        self._out_channels = channels
        self._stride = stride
        self.resblock = residual_block_type(in_channels, channels, stride=stride, **resblock_extra_kwargs)
        self.actblock = nn.ReLU(inplace=True)  # Replace me in subclasses; FIXME check that this doesn't cause an
        # error when computing the gradients
        if stride != 1 or self._in_channels != self._out_channels:
            # Two cases for the use of "projection" shortcuts:
            # - Different fmap size because of striding/downsampling
            # - Different number of channels, because the previous layer isn't a residual block (usually the 1st layer)
            self.shortcut = nn.Sequential(  # Replace me in subclasses
                nn.Conv2d(self._in_channels, self._out_channels, kernel_size=1, stride=self._stride),
                nn.BatchNorm2d(self._out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self._post_init()

    def _post_init(self):
        pass

    def forward(self, x):
        r = self.resblock(x)
        x = self.shortcut(x)
        out = r + x
        out = self.actblock(out)
        return out


class IdentityShortcutPostReLUResidualUnit(ResidualUnit):
    r"""
    A residual unit where the shortcut connection consists of the identity (Option B in the original ResNet paper),
    and the activation after the elem-wise op is ReLU (standard ResNet, as opposed to PreActResNet).
    During downsampling of the fmaps, in the shortcut connection, a 1x1 convolution with stride 2is performed
    (with a BN after it) so that the output shape matches.

    This is usually the default.
    """


class IdentityShortcutDownPadPostReLUResidualUnit(ResidualUnit):
    r"""
    A residual unit where the shortcut connection consists of the identity (Option A in the original ResNet paper),
    and the activation after the elem-wise op consists of ReLU (standard ResNet, as opposed to PreActResNet).
    During downsampling of the fmaps, in the shortcut connection, the image is subsampled (with a stride),
    and additional channels with fmap of zeroes are created so that the output shape matches.
    """
    def _post_init(self):
        if self._stride != 1 or self._in_channels != self._out_channels:
            pad = (self._out_channels - self._in_channels) // 2  # padding (usually self._out_channels // 4
            self.shortcut = LambdaLayer(
                lambda x:
                F.pad(
                    x[:, :, ::self._stride, ::self._stride],  # subsample the image with the stride (usually 2)
                    [0, 0, 0, 0, pad, pad],  # then pad the output tensor: this will add pad*2 channels with only zeroes
                    "constant", 0
                )
            )


class IdentityShortcutPreActResidualUnit(ResidualUnit):
    r"""
    A residual unit where the shortcut connection consists of the identity (Option A in the original ResNet paper),
    and there is no activation after the element-wise operation (Pre-activation ResNet).
    During downsampling of the fmaps, in the shortcut connection,
    """
    def _post_init(self):
        self.actblock = nn.Identity()


class ProjectionShortcutPostReLUResidualUnit(ResidualUnit):
    r"""
    A residual unit where the shortcut connections consist of 1x1 convolutions that act like projections
    (Option C in the original ResNet paper), and the activation after the elem-wise op consists or ReLU
    (standard ResNet, as opposed to PreActResNet).
    Downsampling is implemented by a strided 1x1 convolution that also acts like a projection.
    """
    def _post_init(self):
        self.shortcut = nn.Sequential(  # Do this for both situations, instead of just for strided layer!
            nn.Conv2d(self._in_channels, self._out_channels, kernel_size=1, stride=self._stride),
            nn.BatchNorm2d(self._out_channels)
        )
