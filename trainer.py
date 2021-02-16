from argparse import ArgumentParser, FileType, Namespace
import logging
import time
from typing import List

import torch
from torch import Tensor
import torch.nn as nn
#from torch.nn.modules.loss import _Loss as Loss
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.backends import cudnn

from datasets.cifar10 import CIFAR10
#from models.resnet import ResNet, ResNet20
from utils.statistics_meter import AverageMeter
from utils.argparse import PathType
from resnet_aka import resnet20


args: Namespace


def main():
    # Arguments
    parser = ArgumentParser()

    dataset_opt_group = parser.add_argument_group(title="Dataset options")
    dataset_opt_group.add_argument('--dataset_root', '--dsr', type=str, #PathType(exists=True, dirtype='dir', dash_ok=False),
                                   default='~/datasets',
                                   help="Dataset root folder;"
                                        "invidiual datasets are expected to be in subfolders of this folder")
    dataset_opt_group.add_argument('--dataset', '--ds', type=str, default="CIFAR10",
                                   help="Dataset used")
    dataset_opt_group.add_argument('--batch_size', '--bs', type=int, default=128,
                                   help="Batch size")
    dataset_opt_group.add_argument('--validset', type=str, default='testset',
                                   help="Validation set")

    learning_opt_group = parser.add_argument_group(title="Learning regime options")
    learning_opt_group.add_argument('--learning_rate', '--lr', type=float, default=0.1,
                                    help="Initial learning rate (after startup phase)")
    learning_opt_group.add_argument('--momentum', type=float, default=0.9,
                                    help="Momentum")
    learning_opt_group.add_argument('--weight_decay', type=float, default=1e-4,
                                    help="Weight decay (l2 norm of weights)")
    learning_opt_group.add_argument('--num_epochs', type=int, default=200,
                                    help="Number of epochs in the main training regime")

    compute_opt_group = parser.add_argument_group(title="Compute options group")
    compute_opt_group.add_argument('--device', type=str, default='cpu',
                                   help="Device(s) to use for training")

    logging_opt_group = parser.add_argument_group(title="Logging and printing options")
    logging_opt_group.add_argument('--print_freq', type=int, default=50,
                                   help="Frequency to print progress during train/test; in batches")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Increase verbosity level")  # TODO expand
    global args
    (args, rest) = parser.parse_known_args()
    # parser.add_mutually_exclusive_group(required=True/False) -> group.add_argument etc
    # Sub-parsers: only for sub-commands, not for sub-parsing, because they are selected based on the name of a command...
    lr = args.learning_rate
    momentum = args.momentum
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True  # PyCharm cannot understand this because it is a hack

    model = resnet20().to(device)

    dataset = CIFAR10(args.dataset_root)
    train_loader = dataset.get_train_loader(batch_size=batch_size, shuffle=True, num_workers=4, use_random_crops=True,
                                            use_hflip=True)
    test_loader = dataset.get_test_loader(batch_size=batch_size, shuffle=False, num_workers=4)

    best_prec1 = 0

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)

    for epoch in range(args.num_epochs):
        print(f"Current lr: {optimizer.param_groups[0]['lr']:.5e}")

        train(model, train_loader, criterion, optimizer, epoch, device)
        lr_scheduler.step()

        prec1 = validate(model, test_loader, criterion, device)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if epoch > 0 and epoch % 20 == 0:
            pass  # This was to save the model, but for now let's not use it.


def train(model: nn.Module, train_loader: DataLoader, criterion, optimizer: optim.Optimizer,
          epoch: int, device):
    r"""
    Train for one epoch.
    :param model: The model to train.
    :param train_loader: A DataLoader for the dataset used to train the model
    :param criterion: Criterion to be optimized
    :param optimizer: Optimizer (used to apply zero_grad and step)
    :param epoch: the current epoch
    :param device: The device used for computations.
    :return: Nothing
    """
    print_freq = args.print_freq
    batch_compute_time = AverageMeter()
    data_load_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()  # A bit useless for CIFAR-10, but you never know...
    model.train()
    loop_time = time.time()
    for i, data in enumerate(train_loader):
        inputs, targets = data[0].to(device), data[1].to(device)  # type: Tensor, Tensor
        data_load_time.update(time.time() - loop_time)
        outputs = model(inputs)  # type: Tensor
        loss = criterion(outputs, targets)  # type: Tensor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = outputs.float()
        loss = loss.float()
        prec1, prec5 = accuracy(outputs.data, targets, (1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_compute_time.update(time.time() - loop_time)

        if i % print_freq == 0:
            print(
                f'Train: Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Time {batch_compute_time.val:.2f}\t'
                f'Loss {losses.val:.3f} ({losses.avg:.4f})\t'
                f'Prec@1 {100 * top1.val:.2f} ({100 * top1.avg:.3f})\t'
                f'Prec@5 {100 * top5.val:.2f}'
            )

        loop_time = time.time()
    print(
        f'Train finished\t'
        f'Time {batch_compute_time.avg:.3f}\t'
        f'(Dl {data_load_time.avg:.3f})\t'
        f'Loss {losses.avg:.4f}\t'
        f'Prec@1 {100 * top1.avg:.3f}'
        f'Prec@5 {100 * top5.avg:.3f}'
    )


def validate(model: nn.Module, val_loader: DataLoader, criterion, device):
    print_freq = args.print_freq
    batch_compute_time = AverageMeter()
    data_load_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()  # A bit useless for CIFAR-10, but you never know...
    model.eval()
    loop_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets = data[0].to(device), data[1].to(device)  # type: Tensor, Tensor
            data_load_time.update(time.time() - loop_time)
            outputs = model(inputs)  # type: Tensor
            loss = criterion(outputs, targets)  # type: Tensor

            outputs.float()
            loss.float()

            prec1, prec5 = accuracy(outputs.data, targets, (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            batch_compute_time.update(time.time() - loop_time)

            if i % print_freq == 0:
                print(
                    f'Test: [{i}/{len(val_loader)}]\t'
                    f'Time {batch_compute_time.val:.2f}\t'
                    f'Loss {losses.val:.3f} ({losses.avg:.4f})\t'
                    f'Prec@1 {100 * top1.avg:.3f}\t'
                    f'Prec@5 {100 * top5.avg:.3f}'
                )

            loop_time = time.time()
    print(
        f'Test finished\t'
        f'Time {batch_compute_time.avg:.3f}\t'
        f'(Dload {data_load_time.avg:.3f})\t'
        f'Loss {losses.avg:.4f}\t'
        f'Prec@1 {100 * top1.avg:.3f}'
        f'(Prec@5) {100 * top5.avg:.3f}'
    )
    return top1.avg


def accuracy(outputs: Tensor, targets: Tensor, topk=(1,)) -> List[Tensor]:
    r"""
    Computes the accuracy over the k top predictions for the specified values of k.
    :param outputs:
    :param targets:
    :param topk:
    :return: A list of the top-i accuracy for i in topk. Note that this is not a percentage, but the [0, 1] value
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        corrects = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            corrects_k = corrects[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(corrects_k / batch_size)
        return res


if __name__ == '__main__':
    main()
