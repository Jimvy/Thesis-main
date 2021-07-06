import os
import sys
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import lr_scheduler as topt_lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import cifar
import models
from evaluate import validate
from code.criterion import MultiCriterion, CrossEntropyLossCriterion, HKDCriterion
from code.scheduling import LRSchedulerSequence
from utils.acc import accuracy
from utils.parsing import get_parser, parse_args, args, add_training_args
from utils.statistics_meter import AverageMeter
from utils.tensorboard_logging import get_folder_name


_checkpoint_filename_fmt = None


def get_writer(log_subfolder):
    writer = SummaryWriter(log_subfolder)
    layout = {
        'Prec@1': {
            'prec@1': ['Multiline', ['Prec1/train', 'Prec1/valid']],
        },
        'Losses': {
            'Train': ['Multiline', ['Loss/train']],
            'Valid': ['Multiline', ['Loss/valid']]
        }
    }
    writer.add_custom_scalars(layout)
    return writer


def get_trainer_parser():
    parser = get_parser('Training script for CIFAR datasets')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')

    add_training_args(parser)

    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (per epoch)')
    parser.add_argument('--log-freq', '--lf', default=4, type=int, metavar='N',
                        help="TensorBoard log frequency during training (per epoch)")
    parser.add_argument('--save20', default=False, type=bool,
                        help='Save the best model of every 20 epochs')

    parser.add_argument('--log-dir', '--ld', default='runs', type=str,
                        help="Log folder for TensorBoard")

    return parser


def main():
    global _checkpoint_filename_fmt
    parser = get_trainer_parser()
    args = parse_args(parser)

    if args.evaluate:
        print("Sorry, this script can only be used to train a network; you probably want evaluate.py")
        sys.exit(-1)

    cudnn.benchmark = True

    dataset = cifar.__dict__[args.dataset]('~/datasets', pin_memory=True)

    if args.use_test_set_as_valid:
        train_loader = dataset.get_train_loader(
            args.batch_size, shuffle=True,
            num_workers=args.workers,
        )
        val_loader = dataset.get_test_loader(512, num_workers=args.workers)
    else:
        train_loader, val_loader = dataset.get_train_val_loaders(
            args.batch_size, shuffle=True,
            num_workers=args.workers,
        )  # By default the split is at 90%/10%, so 45k/5k

    model = models.__dict__[args.arch](
        num_classes=dataset.get_num_classes(),
        base_width=args.base_width
    )
    model.cuda()

    # define loss function (criterion)
    criterion = MultiCriterion()
    criterion.add_criterion(CrossEntropyLossCriterion(), "CE")

    # Handling Hinton knowledge distillation
    teacher = None
    if args.distill:
        teacher = models.__dict__[args.teacher_arch](
            num_classes=dataset.get_num_classes(),
            base_width=args.teacher_base_width
        ).cuda()
        if not os.path.isfile(args.teacher_path):
            print(f"No checkpoint found at '{args.teacher_path}'; aborting")
            return
        chkpt = torch.load(args.teacher_path)
        teacher.load_state_dict(chkpt['state_dict'])
        print("Loaded teacher")
        criterion.add_criterion(HKDCriterion(teacher, args.distill_temp), "HKD", weight=args.distill_weight)

    # Define optimizer: mini-batch SGD with momentum
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler1 = topt_lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=args.lr_decay)
    main_lr_scheduler = LRSchedulerSequence(lr_scheduler1)
    if args.use_lr_warmup:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1
        lr_scheduler2 = topt_lr_scheduler.MultiStepLR(
            optimizer,
            gamma=10,
            milestones=[args.lr_warmup_num_epochs]  # First two epochs
        )
        main_lr_scheduler.add_scheduler(lr_scheduler2)

    if args.print_freq < 1:
        args.print_freq = 1
    if args.log_freq < 1:
        args.log_freq = 1

    # Logging-related stuff
    log_subfolder = get_folder_name(args, model, teacher, evaluate_mode=False)
    print(f"Logging into folder {log_subfolder}")
    log_subfolder = os.path.join(args.log_dir, log_subfolder)
    _checkpoint_filename_fmt = os.path.join(log_subfolder, 'model{}.th')
    writer = get_writer(log_subfolder)

    # TODO: add hparams to TensorBoard

    train(train_loader, val_loader, model, criterion, optimizer, main_lr_scheduler, writer)

    # TODO: add precision-recall curve
    if hasattr(writer, "flush"):
        writer.flush()
    writer.close()


def train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, writer):

    best_prec1 = 0
    best_last20_prec1 = 0

    for epoch in range(0, args('epochs')):

        # train for one epoch
        writer.add_scalar("base_learning_rate", optimizer.param_groups[0]['lr'], epoch)
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, writer)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            torch.save({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, _checkpoint_filename_fmt.format(""))

        if epoch % 20 == 0: # Reset, don't need to save it: the next one should _at least_ happen once, hopefully
            best_last20_prec1 = 0
        # remember best prec@1 of the last 20 epochs, and save it
        if args('save20') and prec1 > best_last20_prec1:
            best_last20_prec1 = prec1
            epoch_rounded = ((epoch // 20)+1) * 20
            torch.save({
                'state_dict': model.state_dict(),
                'best_prec1_last20': best_last20_prec1,
                'epoch': epoch,
            }, _checkpoint_filename_fmt.format(f"_epoch{epoch_rounded}"))


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, writer):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_components = {}
    for loss_name in criterion.criterion_names:
        losses_components[loss_name] = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    print_period = (len(train_loader) // args('print_freq')) + 1
    log_period = (len(train_loader) // args('log_freq')) + 1

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs = model(input_var)
        loss = criterion(input_var, outputs, targets)
        loss_map = criterion.prev_ret_map

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

        if i % print_period == print_period-1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'
                  'DL {data_time.val:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        if i % log_period == log_period-1:
            writer.add_scalar("Prec1/train", top1.avg, epoch + i/len(train_loader))
            writer.add_scalar("Loss/train", losses.avg, epoch + i/len(train_loader))
            for loss_name, loss_meter in losses_components.items():
                writer.add_scalar("Loss/train/{}".format(loss_name), loss_meter.avg, epoch + i/len(train_loader))

    print('Epoch: [{0}][done/{1}]\t'
          'Time {batch_time.val:.3f}\t'
          'DL {data_time.val:.3f}\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
              epoch, len(train_loader), batch_time=batch_time,
              data_time=data_time, loss=losses, top1=top1))
    writer.add_scalar("Prec1/train", top1.avg, epoch+1)
    writer.add_scalar("Loss/train", losses.avg, epoch+1)
    for loss_name, loss_meter in losses_components.items():
        writer.add_scalar("Loss/train/{}".format(loss_name), loss_meter.avg, epoch+1)


if __name__ == '__main__':
    main()
