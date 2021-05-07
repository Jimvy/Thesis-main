import sys

import numpy as np
import torch

__all__ = ['error', 'polar_to_rect', 'rect_to_polar', 'to_radians', 'to_rect']


def error(s: str):
    print(s, file=sys.stderr)
    sys.exit(-1)


def polar_to_rect(norm, angle):
    r"""
    Polar to rectangular
    :param norm: B values for the norm (distance from 0)
    :param angle: B values for the angle (measured from X axis)
    :return: a tuple of 2 tensors of B values each, X and Y.
    """
    return norm * torch.cos(angle), norm * torch.sin(angle)


def rect_to_polar(x, y):
    r"""
    Rectangular to polar
    :param x: B values for X coord
    :param y: B values for Y coord
    :return: tuple of 2 tensors of B values each, norm and angles
    """
    return torch.sqrt(x**2 + y**2), torch.atan2(y, x)


def to_radians(x):
    return x * (2*np.pi / 360)


def to_rect(x, y, norm, angle):
    cnt = 0
    for xx in [x, y, norm, angle]:
        if xx is not None:
            cnt += 1
    assert(cnt == 2)
    if x is not None and y is not None:
        return x, y
    if norm is not None and angle is not None:
        return polar_to_rect(norm, to_radians(angle))
    if norm is not None and x is not None:
        y = torch.sqrt(norm ** 2 - x ** 2)
        return x, y
    if norm is not None and y is not None:
        x = torch.sqrt(norm ** 2 - y ** 2)
        return x, y
    if x is not None and angle is not None:
        y = torch.tan(to_radians(angle)) * x
        return x, y
    if y is not None and angle is not None:
        x = y / torch.tan(to_radians(angle))
        return x, y
    return None
