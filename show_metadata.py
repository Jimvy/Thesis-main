import argparse
import os
import sys

import torch

TRANS = {
    'lr': 'learning rate',
    'wd': 'weight decay',
    'lr_decay': 'learning rate decay',
    'bs': 'batch size',
}
print_bak = print

def main():
    global args
    parser = argparse.ArgumentParser('Show metadata inside a checkpoint')
    parser.add_argument('chkpt', metavar='SAVED_MODEL',
                        help='Path to model checkpoint whose metadata have to be shown')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--follow', '-f', type=bool, default=True,
                        help='Follow recursively paths to other models and display their parameters too')
    args = parser.parse_args()
    if not show_checkpoint(args.chkpt):
        sys.exit(-1)

def show_checkpoint(model_chkpt_path, indent_level=0):
    def print(s, **kwargs):
        print_bak(('\t'*indent_level) + s, **kwargs)
    if not os.path.isfile(model_chkpt_path):
        print(f"No checkpoint found at '{model_chkpt_path}'", file=sys.stderr)
        return False
    print(f"Loading checkpoint '{model_chkpt_path}'", file=sys.stderr)
    chkpt = torch.load(model_chkpt_path)

    dataset_params = chkpt['dataset']
    epoch = chkpt['epoch']
    prec1 = chkpt.get('prec1', None)
    best_prec1 = chkpt.get('best_prec1', None)
    best_prec1_last20 = chkpt.get('best_prec1_last20', None)
    arch_params = chkpt['arch']
    train_params = chkpt['train_params']
    distill_params = train_params.get('distill', None)

    if arch_params['arch'].startswith('resnet'):
        arch_params['arch'] = 'ResNet' + arch_params['arch'][6:]
    elif arch_params['arch'].startswith('vgg'):
        arch_params['arch'] = arch_params['arch'].upper()
    best_prec1_s = f", best prec1 {best_prec1}" if best_prec1 != prec1 else ''
    best_prec1_last20_s = f", best prec1 last 20 {best_prec1_last20}" if best_prec1_last20 else ''
    print(f"{arch_params['arch']}-{arch_params['base_width']}, checkpoint at epoch {epoch}, prec1 {prec1}{best_prec1_last20_s}{best_prec1_s}")
    print(f"Trained on {dataset_params['name']}, used {'test' if dataset_params['test_set_as_valid'] else 'valid'} set as validation set.")
    print("Trained with the following params:")
    for k, v in train_params.items():
        if k != 'distill' and k != 'warmup':
            print(f"\t{TRANS.get(k, k)} = {v}")
        if k == 'warmup':
            print(f"\twarmup during {v['num_epochs']}")
    if distill_params:
        print("Trained with distillation:")
        print(f"\tTemperature = {distill_params['temp']}")
        print(f"\tWeight = {distill_params['weight']}")
        if os.path.isfile(distill_params['teacher_path']):
            teacher_path = distill_params['teacher_path']
        elif os.path.isfile(distill_params['teacher_path_rel']):
            teacher_path = distill_params['teacher_path_rel']
        elif os.path.isfile(distill_params['teacher_path_abs']):
            teacher_path = distill_params['teacher_path_abs']
        else:
            teacher_path = distill_params['teacher_path']
        print(f"Teacher at {teacher_path}")
        if args.follow:
            if not show_checkpoint(teacher_path, indent_level+1):
                return False
    return True

if __name__ == '__main__':
    main()

