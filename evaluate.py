"""
Evaluation-related functions.


"""
import time

import torch

from parsing import args
from utils.acc import accuracy
from utils.statistics_meter import AverageMeter


def validate(val_loader, model, criterion, epoch, writer=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_components = {}
    for loss_name in criterion.criterion_names:
        losses_components[loss_name] = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            targets = targets.cuda()
            inputs = inputs.cuda()

            if args('half'):
                inputs = inputs.half()

            # compute output
            outputs = model(inputs)
            loss = criterion(inputs, outputs, targets)
            loss_map = criterion.prev_ret_map

            outputs = outputs.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            for loss_name, loss_val in loss_map.items():
                losses_components[loss_name].update(loss_val)
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if writer:
            writer.add_scalar("Prec1/valid", top1.avg, epoch)
            writer.add_scalar("Loss/valid", losses.avg, epoch)
            for loss_name, loss_meter in losses_components.items():
                writer.add_scalar("Loss/valid/{}".format(loss_name), loss_meter.avg, epoch)

    print(f"Valid: Prec1 {top1.avg:.3f} \t (Time: {batch_time.avg:.3f}, Loss: {losses.avg:.4f})")

    return top1.avg

