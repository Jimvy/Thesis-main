import os
import sys
from datetime import datetime

_FOLDER_IGNORED_ARGS_TRAIN = [
    'arch', 'workers', 'log_freq', 'print_freq', 'momentum',
    'epochs', 'teacher_path', 'log_dir', 'save20',
]
_FOLDER_INCLUDED_ARGS = [('bs', 'batch_size'), ('lr', 'lr'), ('lr_dec', 'lr_decay'), ('wd', 'weight_decay')]
_FOLDER_IGNORED_ARGS_TEST = [
    'batch_size', 'arch', 'workers', 'log_freq', 'print_freq', 'momentum',
    'epochs', 'teacher_path', 'log_dir', 'save20',
]


def get_folder_name(args, main_model, teacher,
                    evaluate_mode=False):
    FOLDER_IGNORED_ARGS = _FOLDER_IGNORED_ARGS_TEST if evaluate_mode else _FOLDER_IGNORED_ARGS_TRAIN
    arg_keys = sorted(vars(args).keys())
    attrs = []
    attrs.append(args.dataset)
    arg_keys.remove('dataset')
    attrs.append(main_model.get_model_name())
    arg_keys.remove('arch')
    arg_keys.remove('base_width')
    if evaluate_mode:
        print("WARNING, you are doing something extremely disgusting here, I don't agree with it", file=sys.stderr)
        arg_keys.remove("no_reuse_folder")
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
    if not evaluate_mode:
        for (arg_key_print, arg_key_name) in _FOLDER_INCLUDED_ARGS:
            attrs.append(f'{arg_key_print}={getattr(args, arg_key_name)}')
            arg_keys.remove(arg_key_name)
    if hasattr(args, 'use_lr_warmup'):
        if args.use_lr_warmup:
            attrs.append(f'use_warmup_num_epochs={args.lr_warmup_num_epochs}')
        arg_keys.remove('use_lr_warmup')
        arg_keys.remove('lr_warmup_num_epochs')
    if args.use_test_set_as_valid:
        attrs.append(f"validation=test_set")
    arg_keys.remove('use_test_set_as_valid')
    for arg_key in FOLDER_IGNORED_ARGS:
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
    if not evaluate_mode:
        attrs.append('gpu{}'.format(os.environ.get('CUDA_VISIBLE_DEVICES', 'all')))
    if args.comment:
        attrs.append(args.comment)
    return '_'.join(attrs)
