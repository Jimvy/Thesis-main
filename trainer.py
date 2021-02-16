from typing import List

import time

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.backends import cudnn

from datasets.cifar10 import CIFAR10
from models.resnet import ResNet, ResNet20
from utils.statistics_meter import AverageMeter


def main():
    # Arguments
    lr = 0.1
    momentum = 0.9
    batch_size = 128
    weight_decay = 1e-4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True  # PyCharm cannot understand this bcause it is a hack

    model = ResNet20().to(device)

    dataset = CIFAR10("~/datasets")
    train_loader = dataset.get_train_loader(batch_size=batch_size, shuffle=True, num_workers=4, use_random_crops=True, use_hflip=True)
    test_loader = dataset.get_test_loader(batch_size=batch_size, shuffle=False, num_workers=4)

    best_prec1 = 0

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)

    for epoch in range(200):
        print(f"Current lr: {optimizer.param_groups[0]['lr']:.5e}")

        train(model, train_loader, criterion, optimizer, epoch, device)
        lr_scheduler.step()

        prec1 = validate(model, test_loader, criterion, device)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if epoch > 0 and epoch % 20 == 0:
            pass  # This was to save the model, but for now let's not use it.


def train(model: nn.Module, train_loader: DataLoader, criterion: Loss, optimizer: optim.Optimizer,
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
    print_freq = 50
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
                f'Time {batch_compute_time.val:.3f} ({batch_compute_time.avg:.3f})\t'
                f'(Data load {data_load_time.val:.3f} ({data_load_time.avg:.3f}))\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {100*top1.val:.3f} ({100*top1.avg:.3f})'
            )

        loop_time = time.time()


def validate(model: nn.Module, val_loader: DataLoader, criterion: Loss, device):
    print_freq = 50
    batch_compute_time = AverageMeter()
    data_load_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
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

            prec1 = accuracy(outputs.data, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_compute_time.update(time.time() - loop_time)

            if i % print_freq == 0:
                print(
                    f'Test: [{i}/{len(val_loader)}]\t'
                    f'Time {batch_compute_time.val:.3f} ({batch_compute_time.avg:.3f})\t'
                    f'(Data load {data_load_time.val:.3f} ({data_load_time.avg:.3f}))\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Prec@1 {100*top1.val:.3f} ({100*top1.avg:.3f})'
                )

            loop_time = time.time()
    print(f' * Prec@1 {top1.avg:.3f}')
    return top1.avg


def accuracy(outputs: Tensor, targets: Tensor, topk=(1, )) -> List[Tensor]:
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
