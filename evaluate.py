"""
Evaluation-related functions.


"""
import argparse
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter

import cifar
from code.criterion import MultiCriterion, CrossEntropyLossCriterion, HKDCriterion
from utils.acc import accuracy
from utils.checkpoint import load_dataset_args_from_checkpoint_or_args, load_model_from_checkpoint_or_args,\
    load_teacher_from_checkpoint_or_args, make_args_fun
from utils.parsing import get_parser, parse_args
from utils.statistics_meter import AverageMeter
from utils.tensorboard_logging import get_folder_name


_critname_to_crit = {'CE': CrossEntropyLossCriterion, 'HKD': HKDCriterion}


def get_evaluate_parser():
    parser = get_parser('Evaluation script for CIFAR datasets')

    parser.add_argument('chkpt', metavar='SAVED_MODEL',
                        help='Path to model checkpoint to evaluate')

    parser.add_argument('-b', '--batch-size', '--bs', default=256, type=int,
                        metavar='N', help='mini-batch size')

    parser.add_argument('--log-dir', '--ld', default='evals', type=str,
                        help='Log folder for TensorBoard. Ignored for the better if reusing original training folder')
    parser.add_argument('--no-reuse-folder', default=False, action='store_true',
                        help="Don't reuse original training folder name as destination folder."
                             "Seriously, don't do this, the folder name is disgusting.")
    parser.add_argument('--eval-distrib', action='store_true', default=False,
                        help='Evaluate the model and add output distributions to the log folder.')

    return parser


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
    args_fun = make_args_fun(args)

    cudnn.benchmark = True

    model_chkpt_path = args.chkpt
    if not os.path.isfile(model_chkpt_path):
        print(f"No checkpoint found at '{args.chkpt}'", file=sys.stderr)
        sys.exit(-1)
    print("Loading checkpoint '{}'".format(model_chkpt_path), file=sys.stderr)
    chkpt = torch.load(model_chkpt_path)
    best_prec1 = chkpt['best_prec1'] if 'best_prec1' in chkpt else chkpt['best_prec1_last20']
    prec1 = chkpt['prec1'] if 'prec1' in chkpt else best_prec1
    epoch = chkpt['epoch'] if 'epoch' in chkpt else 200
    print(f"Loaded checkpoint, epochs up to {epoch}, accuracy {prec1}/{best_prec1}", file=sys.stderr)

    # Dataset
    dataset_name, use_test_set_as_valid = load_dataset_args_from_checkpoint_or_args(chkpt, args)
    args.dataset = dataset_name  # dirty fix
    dataset = cifar.__dict__[dataset_name]('~/datasets', pin_memory=True)
    num_classes = dataset.get_num_classes()
    if use_test_set_as_valid:
        val_loader = dataset.get_test_loader(512, num_workers=args.workers)
    else:
        _, val_loader = dataset.get_train_val_loaders(
            args.batch_size, shuffle=True,
            num_workers=args.workers,
        )  # By default the split is at 90%/10%, so 45k/5k

    # Model
    model = load_model_from_checkpoint_or_args(chkpt, args_fun, num_classes=num_classes)

    criterion = MultiCriterion()
    criterion.add_criterion(CrossEntropyLossCriterion(), "CE")

    if args.distill:
        teacher = load_teacher_from_checkpoint_or_args(args, chkpt=chkpt, num_classes=num_classes)
        print(f"Loaded teacher", file=sys.stderr)
        criterion.add_criterion(HKDCriterion(teacher, args.distill_temp), "HKD", weight=args.distill_weight)
    else:
        teacher = None

    # Logging-related stuff
    if args.no_reuse_folder:
        log_subfolder = os.path.join(args.log_dir, get_folder_name(args, model, teacher, evaluate_mode=True))
    else:
        log_subfolder = os.path.dirname(model_chkpt_path)
    if args.eval_distrib:
        print(f"Logging into folder {log_subfolder}", file=sys.stderr)

    validate(val_loader, model, criterion, epoch, evaluate_output_distrib=False)

    if args.eval_distrib:  # TODO if other kinds of evaluation, add them here
        writer = get_writer(log_subfolder)

        # TODO: add hparams to TensorBoard

        evaluate(val_loader, model, writer, epoch, double=True)

        # TODO: add precision-recall curve
        if hasattr(writer, "flush"):
            writer.flush()
        writer.close()


def evaluate(val_loader, model, writer, epoch, double=False):
    model.eval()
    with torch.no_grad():
        stats_lists = {
            'output/target': [],
            'output/predicted': [],
            'output/correct': [],
            'output/target_wrong': [],
            'output/predicted_wrong': [],
            'softmax/target': [],
            'softmax/predicted': [],
            'softmax/correct': [],
            'softmax/target_wrong': [],
            'softmax/predicted_wrong': [],
        }
        for i, (inputs, targets) in enumerate(val_loader):
            targets = targets.cuda()  # B
            B = targets.shape[0]
            inputs = inputs.cuda()  # Bx3x32x32
            raw_outputs = model(inputs)  # BxC
            softmax_outputs = F.softmax(raw_outputs, dim=1)  # BxC
            raw_max_outputs, predicteds = torch.max(raw_outputs, dim=1)  # Bx1, Bx1
            idx_correct = (predicteds == targets)  # B
            idx_incorrect = ~idx_correct
            # https://stackoverflow.com/questions/61096522/pytorch-tensor-advanced-indexing
            ro_idxtt = raw_outputs[range(B), targets]
            ro_idxtt_incorrect = ro_idxtt[idx_incorrect]
            ro_idxpp_incorrect = raw_max_outputs[idx_incorrect]
            ro_idx_correct = ro_idxtt[idx_correct]
            stats_lists['output/correct'].append(ro_idx_correct)
            stats_lists['output/target'].append(ro_idxtt)
            stats_lists['output/predicted'].append(raw_max_outputs)
            stats_lists['output/target_wrong'].append(ro_idxtt_incorrect)
            stats_lists['output/predicted_wrong'].append(ro_idxpp_incorrect)
            # softmax
            sftm_idxtt = softmax_outputs[range(B), targets]
            sftm_idxpp = softmax_outputs[range(B), predicteds]
            sftm_idxtt_incorrect = sftm_idxtt[idx_incorrect]
            sftm_idxpp_incorrect = sftm_idxpp[idx_incorrect]
            sftm_idx_correct = sftm_idxtt[idx_correct]
            stats_lists['softmax/correct'].append(sftm_idx_correct)
            stats_lists['softmax/target'].append(sftm_idxtt)
            stats_lists['softmax/predicted'].append(sftm_idxpp)
            stats_lists['softmax/target_wrong'].append(sftm_idxtt_incorrect)
            stats_lists['softmax/predicted_wrong'].append(sftm_idxpp_incorrect)
        for tag, l in stats_lists.items():
            writer.add_histogram(tag, torch.cat(l), epoch)
            if double:
                writer.add_histogram(tag, torch.cat(l), epoch+1)


def validate(val_loader, model, criterion, epoch, writer=None, evaluate_output_distrib=False):
    """
    Run evaluation
    if evaluate_output_distrib is True, it also executes evaluate(val_loader, model, writer, epoch). writer should be non-None.
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

    print(f"Valid: Prec1 {top1.avg:.3f} \t (Time: {batch_time.avg:.3f}, Loss: {losses.avg:.4f})", file=sys.stderr)

    if evaluate_output_distrib and writer:
        evaluate(val_loader, model, writer, epoch)

    return top1.avg


if __name__ == '__main__':
    main()

