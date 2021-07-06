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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve',
    )
    add_dataset_args(parser)

    add_arch_args(parser)

    add_distill_args(parser)

    parser.add_argument('--comment', type=str, help='Commentary on the run')

    return parser


def add_dataset_args(parser, with_default_dataset=True):
    if with_default_dataset:
        parser.add_argument('--dataset', '--ds', default='CIFAR10',
                            choices=["CIFAR10", "CIFAR100", "CIFAR100Coarse"],
                            help="Dataset to use")
    else:
        parser.add_argument('--dataset', '--ds',
                            choices=["CIFAR10", "CIFAR100", "CIFAR100Coarse"],
                            help="Dataset to use")
    parser.add_argument('--use-test-set-as-valid', action='store_true',
                        help='Use test set as validation set, and the full train set as train set, instead of the 5k/45k split')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', '--bs', default=128, type=int,
                        metavar='N', help='mini-batch size')


def add_arch_args(parser):
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                        choices=_model_names,
                        help='model architecture: ' + ' | '.join(_model_names) +
                             ' (default: resnet32)')
    parser.add_argument('--base-width', metavar='WIDTH', default=16, type=int,
                        help='width of the base layer')


def add_training_args(parser):
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate'
                                           '\nNote that for ResNet-112/1202 it is 1e-2')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--use-lr-warmup', action='store_true',
                        help="Use learning scheduler 2 to warmup the learning rate")
    parser.add_argument('--lr-warmup-num-epochs', type=int, default=2,
                        help='Number of epochs for the warmup, if set')
    parser.add_argument('--lr-decay', default=0.1, type=float,
                        help='Learning rate decay factor')


def add_distill_args(parser):
    parser.add_argument('--distill', action='store_true',
                        help='Specify distillation parameters')
    parser.add_argument('--distill-weight', type=float,
                        help='Distillation weight')
    parser.add_argument('--distill-temp', type=float,
                        help='Distillation temperature')
    parser.add_argument('--teacher-arch', type=str,
                        help='Teacher architecture')
    parser.add_argument('--teacher-base-width', type=int,
                        help='Teacher architecture base width')
    parser.add_argument('--teacher-path', type=str, metavar='PATH',
                        help='Teacher model checkpoint path')


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

