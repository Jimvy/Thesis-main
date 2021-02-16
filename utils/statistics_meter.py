r"""
Classes to handle statistics about values.
"""

import torch

from .math import sqrt

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        r"""
        Reset the meter.
        :return: Nothing
        """
        self.__init__()  # Disgusting, but PyCharm prefers this...

    def update(self, val, n=1):
        r"""
        Update the meter with a new datapoint, :code:`val`.
        :param val: A new value
        :param n: The number of times this value should be added to the meter (default: 1)
        :return: Nothing
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageStddevMeter(AverageMeter):
    def __init__(self):
        super(AverageStddevMeter, self).__init__()
        self.sum_squared = 0
        self.stddev = 0  # Technically it should rather be infinity, but anyway.

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        super(AverageStddevMeter, self).update(val, n)
        self.sum_squared += n * val**2
        self.stddev = sqrt((self.sum_squared / self.count) - self.avg**2)
