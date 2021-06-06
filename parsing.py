"""
Parsing-related functions.

Contains the common parts of the parsing for trainer.py and eval.py

"""
import argparse
import os

import models
import cifar

_model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and (name.startswith("resnet") or name.startswith("vgg"))
    and callable(models.__dict__[name])
)


def get_parser(description='Proper ResNets for CIFAR10 in pytorch'):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', '--ds', default='CIFAR10',
                        choices=["CIFAR10", "CIFAR100", "CIFAR100Coarse"],
                        help="Dataset to use")
    parser.add_argument('--use-test-set-as-valid', action='store_true',
                        help='Use test set as validation set, and the full train set as train set, instead of the 5k/45k split')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                        choices=_model_names,
                        help='model architecture: ' + ' | '.join(_model_names) +
                        ' (default: resnet32)')
    parser.add_argument('--base-width', metavar='WIDTH', default=16, type=int,
                        help='width of the base layer')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision (16-bit)')

    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', '--bs', default=128, type=int,
                        metavar='N', help='mini-batch size')

    parser.add_argument('--log-dir', '--ld', default='runs', type=str,
                        help="Log folder for TensorBoard")

    parser.add_argument('--comment', type=str, help='Commentary on the run')

    return parser

def parse_args(parser):
    """
    Parses the arguments of the program (sys.argv) by using the provided parser.
    #For safety reasons, the resulting arguments are not returned; instead, use the args function.
    """
    global _args
    _args = parser.parse_args()
    return _args

def args(field_name):
    """
    Returns the value of args.field_name.
    Useful to use as a global way to get arguments instead of passing the arguments always.
    """
    global _args
    return getattr(_args, field_name)

