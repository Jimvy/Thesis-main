"""
Script to update the information contained in a .th file

Full structure of a saved model:
{
    "state_dict": model.state_dict(),
    "dataset": {
        "name": "CIFAR100" etc, the dataset the model was trained on,
        "test_set_as_valid": boolean, determines if the valid set was the test set, or the 4k split of the train set.
                                 Also determines the training set used.
    }
    "epoch": int, epoch at which the model was saved,
    "prec1": float, accuracy of this saved model,
    "best_prec1": float, accuracy of the best model so far, usually should be the accuracy of this model,
                  over all the training.
    "best_prec1_last20": float, accuracy of the best model in the last 20 epochs,
                         usually should be the accuracy of this model over the 20 nearing epochs.
    "arch": either
        - a string, like "ResNet32-32" or "VGG-19", usually the name the model gives itself. Should be apt to construct the model back.
        - or a dict: {
            "arch": the short string name provided at the time,
            "base_width": base width
          }
    "train_params": {
        "lr": 0.1, float,
        "momentum": 0.9, float,
        "wd": 1e-4, float, weight decay,
        "warmup": if present, has warmup, and contains {
            "num_epochs": int, number of epochs
        },
        "lr_decay": 0.1, float,
        "bs": int, batch size,
        "distill": if present, indicates that the model was trained with knowledge distillation, and contains {
            "weight": float,
            "temp": temprature,
            "teacher_path": path of the teacher, as specified in the command line.
            "teacher_path_rel": relative path to the teacher, from the location of the saved model.
            "teacher_path_abs": absolute path to the teacher.
            In theory, knowing the teacher save point should be enough to know its architecture.
        }
    }
}
Not present:
- color jitter, as it is unused
- number of worker threads in the data loaders
- print_freq, log_freq
- save20, log_dir
- evaluate, resume
"""
import argparse
import os
import sys

import torch

import cifar
from criterion import MultiCriterion, CrossEntropyLossCriterion
from evaluate import validate
import models
from parsing import parse_args, add_arch_args, add_dataset_args, add_distill_args, add_training_args


def get_update_parser():
    parser = argparse.ArgumentParser(
        description='Script for updating information on a model checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('chkpt', metavar='SAVED_MODEL',
                        help='Path to model checkpoint the information of which we\'ll update')
    parser.add_argument('--validate', action='store_true',
                        help='Run the validation of the model to estimate its accuracy.')
    parser.add_argument('--epoch', type=int,
                        help='The epoch at which this model was saved. Or at least, we think it was saved')

    add_arch_args(parser)
    add_dataset_args(parser, with_default_dataset=False)
    add_distill_args(parser)
    add_training_args(parser)
    return parser


def update_with_warning(chkpt, args, varname):
    if varname in chkpt and hasattr(args, varname) and getattr(args, varname) and chkpt[varname] != getattr(args, varname):
        print(f"WARNING: replacing {chkpt[varname]} with {getattr(args, varname)} as {varname}")
    if hasattr(args, varname) and getattr(args, varname):
        chkpt[varname] = getattr(args, varname)


def main():
    parser = get_update_parser()
    args = parse_args(parser)

    model_chkpt_path = args.chkpt
    if not os.path.isfile(model_chkpt_path):
        print(f"No checkpoint found at '{model_chkpt_path}'")
        sys.exit(-1)
    print(f"Loading checkpoint '{model_chkpt_path}'")
    chkpt = torch.load(model_chkpt_path)

    if 'state_dict' not in chkpt:
        print(f"ERROR: No state_dict found in this checkpoint; probably a mistake, aborting")
        sys.exit(-1)

    present_keys = chkpt.keys()
    print(f"Checkpoint has keys: {present_keys}")
    copy_old_chkpt = {}
    for it in present_keys:
        if it != 'state_dict':
            print(f"checkpoint[{it}] = {chkpt[it]}")
        if it not in ['state_dict', 'epoch', 'best_prec1', 'best_prec1_last20']:
            print(f"WARNING: the checkpoint has an unknown variable called {it} with value {chkpt[it]}")
            copy_old_chkpt[it] = chkpt[it]

    chkpt['dataset'] = {
        'name': args.dataset,
        'test_set_as_valid': args.use_test_set_as_valid,
    }
    update_with_warning(chkpt, args, 'epoch')

    chkpt['arch'] = {
        'arch': args.arch,
        'base_width': args.base_width,
    }

    chkpt['train_params'] = {
        'lr': args.lr,
        'momentum': args.momentum,
        'wd': args.weight_decay,
        'lr_decay': args.lr_decay,
        'bs': args.batch_size,
    }
    if args.use_lr_warmup:
        chkpt['train_params']['warmup'] = {'num_epochs': args.lr_warmup_num_epochs}

    if args.distill:
        teacher_path = args.teacher_path
        if not os.path.isfile(teacher_path):
            print(f"WARNING: incorrect path for the teacher, no model at the path specified")
        teacher_rel_path = os.path.relpath(teacher_path, os.path.dirname(model_chkpt_path))
        teacher_abs_path = os.path.abspath(teacher_path)
        chkpt['train_params']['distill'] = {
            'weight': args.distill_weight,
            'temp': args.distill_temp,
            'teacher_path': teacher_path,
            'teacher_path_rel': teacher_rel_path,
            'teacher_path_abs': teacher_abs_path,
        }

    if args.validate:
        dataset = cifar.__dict__[args.dataset]('~/datasets', pin_memory=True)
        if args.use_test_set_as_valid:
            val_loader = dataset.get_test_loader(512, num_workers=args.workers)
        else:
            _, val_loader = dataset.get_train_val_loaders(
                512, shuffle=True, num_workers=args.workers,
                use_color_jitter=args.use_color_jitter
            )
        model = models.__dict__[args.arch](
            num_classes=dataset.get_num_classes(),
            base_width=args.base_width,
        )
        model.cuda()
        model.load_state_dict(chkpt['state_dict'])
        criterion = MultiCriterion()
        criterion.add_criterion(CrossEntropyLossCriterion(), "CE")
        prec1 = validate(val_loader, model, criterion, 42)
        chkpt['prec1'] = float(f"{prec1:.2f}")

    torch.save(chkpt, model_chkpt_path)
    if copy_old_chkpt:
        print(f"Saving a copy of the old fields of the checkpoint too")
        torch.save(chkpt, model_chkpt_path + ".update.bak")


if __name__ == '__main__':
    main()
