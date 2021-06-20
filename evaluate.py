"""
Evaluation-related functions.


"""
from datetime import datetime
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter

import models
import cifar
from criterion import MultiCriterion, CrossEntropyLossCriterion, HKDCriterion
from parsing import get_parser, parse_args, args
from utils.acc import accuracy
from utils.statistics_meter import AverageMeter


_critname_to_crit = {'CE': CrossEntropyLossCriterion, 'HKD': HKDCriterion}
_FOLDER_INCLUDED_ARGS = []
_FOLDER_IGNORED_ARGS = ['batch_size', 'arch', 'workers', 'resume', 'log_freq', 'print_freq', 'momentum', 'start_epoch', 'epochs', 'teacher_path', 'log_dir']

def get_evaluate_parser():
    parser = get_parser('Evaluation script for CIFAR datasets')

    parser.add_argument('chkpt', metavar='SAVED_MODEL',
                        help='Path to model checkpoint to evaluate')

    parser.add_argument('-b', '--batch-size', '--bs', default=256, type=int,
                        metavar='N', help='mini-batch size')

    parser.add_argument('--log-dir', '--ld', default='evals', type=str,
                        help='Log folder for TensorBoard')

    return parser


def get_folder_name(args, main_model, teacher):
    arg_keys = sorted(vars(args).keys())
    attrs = []
    attrs.append(args.dataset)
    arg_keys.remove('dataset')
    attrs.append(main_model.get_model_name())
    arg_keys.remove('arch')
    arg_keys.remove('base_width')
    print(args.chkpt)
    print(args.chkpt.replace('/', '_'))
    arg_keys.remove("chkpt")
    attrs.append(f"chkpt={args.chkpt.replace('/', '_')}")
    if args.distill:
        attrs.append("distill")
        attrs.append(teacher.get_model_name())
        attrs.append(f"temp={args.distill_temp}")
        attrs.append(f"weight={args.distill_weight}")
        arg_keys.remove("distill")
        arg_keys.remove("distill_temp")
        arg_keys.remove("distill_weight")
        arg_keys.remove("teacher_arch")
        arg_keys.remove("teacher_base_width")
    for (arg_key_print, arg_key_name) in _FOLDER_INCLUDED_ARGS:
        attrs.append(f'{arg_key_print}={getattr(args, arg_key_name)}')
        arg_keys.remove(arg_key_name)
    if args.use_test_set_as_valid:
        attrs.append(f"validation=test_set")
    arg_keys.remove('use_test_set_as_valid')
    for arg_key in _FOLDER_IGNORED_ARGS:
        if arg_key in arg_keys:
            arg_keys.remove(arg_key)
    for arg_key in arg_keys:
        arg_val = getattr(args, arg_key)
        if arg_val is not None and arg_val is not False and arg_key != 'comment':
            if arg_val == True:
                attrs.append(f'{arg_key}')
            else:
                attrs.append(f'{arg_key}={arg_val}')
            arg_keys.remove(arg_key)
    attrs.append('{}'.format(datetime.now().strftime('%b%d_%H-%M-%S')))
    if args.comment:
        attrs.append(args.comment)
    return '_'.join(attrs)


def get_writer(log_subfolder):
    writer = SummaryWriter(log_subfolder)
    layout = {
        #
    }
    writer.add_custom_scalars(layout)
    return writer


def main():
    parser = get_evaluate_parser()
    args = parse_args(parser)

    cudnn.benchmark = True

    dataset = cifar.__dict__[args.dataset]('~/datasets', pin_memory=True)

    if args.use_test_set_as_valid:
        train_loader = dataset.get_train_loader(
            args.batch_size, shuffle=True,
            num_workers=args.workers,
            use_color_jitter=args.use_color_jitter
        )
        val_loader = dataset.get_test_loader(512, num_workers=args.workers)
    else:
        train_loader, val_loader = dataset.get_train_val_loaders(
            args.batch_size, shuffle=True,
            num_workers=args.workers,
            use_color_jitter=args.use_color_jitter
        )  # By default the split is at 90%/10%, so 45k/5k

    model = models.__dict__[args.arch](
        num_classes=dataset.get_num_classes(),
        base_width=args.base_width
    )
    model.cuda()

    model_chkpt_path = args.chkpt
    if os.path.isfile(model_chkpt_path):
        print("Loading checkpoint '{}'".format(model_chkpt_path))
        chkpt = torch.load(model_chkpt_path)
        model.load_state_dict(chkpt['state_dict'])
        best_prec1 = chkpt['best_prec1'] if 'best_prec1' in chkpt else chkpt['best_prec1_last20']
        best_prec_last_epoch = chkpt['epoch'] if 'epoch' in chkpt else 200
        print(f"Loaded checkpoint, epochs up to {best_prec_last_epoch}, accuracy {best_prec1}")
    else:
        print(f"No checkpoint found at '{args.chkpt}'")
        sys.exit(-1)

    criterion = MultiCriterion()
    criterion.add_criterion(CrossEntropyLossCriterion(), "CE")

    if args.distill:
        teacher = models.__dict__[args.teacher_arch](
            num_classes=dataset.get_num_classes(),
            base_width=args.teacher_base_width
        ).cuda()
        if not os.path.isfile(args.teacher_path):
            print(f"No teacher checkpoint found at '{args.teacher_path}'; aborting")
        chkpt = torch.load(args.teacher_path)
        teacher.load_state_dict(chkpt['state_dict'])
        print(f"Loaded teacher")
        criterion.add_criterion(HKDCriterion(teacher, args.distill_temp), "HKD", weight=args.distill_weight)
    else:
        teacher = None

    if args.half:
        model.half()
        criterion.half()

    # Logging-related stuff
    #log_subfolder = os.path.join(args.log_dir, get_folder_name(args, model, teacher))
    #writer = get_writer(log_subfolder)

    # TODO: add hparams to TensorBoard

    validate(val_loader, model, criterion, 42)

    # TODO: add precision-recall curve
    # if hasattr(writer, "flush"):
    #     writer.flush()
    # writer.close()


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

if __name__ == '__main__':
    main()
