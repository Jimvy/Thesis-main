"""
Utility functions to help with loading models from checkpoints
"""
import os
import sys

import torch

import cifar
import models


def make_args_fun(args):
    """
    Creates a function that takes a field name as a string, and returns the value associated with this field in the Namespace.
    This is used by some functions in this module. Essentially the same as getattr.
    :param args: arguments to the program; should be a Namespace or something that supports getattr. Basically anything.
    :return: a function.
    """
    return lambda field_name: getattr(args, field_name)


def make_args_teacher_fun(args):
    """
    Creates a function that takes a field name for a general model, and returns the value in args for a teacher.
    Essentially, does the same stuff as getattr, but with "teacher_" as a prefix.
    :param args: arguments to the program.
    :return: a function.
    """
    def f(field_name):
        if field_name in ['arch', 'base_width']:
            return getattr(args, 'teacher_' + field_name)
    return f


def make_args_teacher_fun_from_fun(args_fun):
    """
    Meta: creates a new function from a function. Argument should be returned by "make_args_fun",
    return value behaves like "make_args_teacher_fun".
    :param args_fun: a function like getattr
    :return: a function like getattr, but which adds 'teacher_' as a prefix
    """
    def f(field_name):
        if field_name in ['arch', 'base_width']:
            return args_fun('teacher_' + field_name)
        return args_fun(field_name)
    return f


def load_dataset_from_checkpoint_or_args(chkpt, args):
    r"""
    Loads a dataset from the given checkpoint, or from the args if some arguments are missing in the checkpoint.
    :param chkpt: a checkpoint.
    :param args: a argparse.Namespace. Sorry for the inconvenience.
    :return: something that I have yet to determine
    """


def load_dataset_args_from_checkpoint_or_args(chkpt, args_fun):
    r"""
    Loads dataset arguments from the given checkpoint, or from the args if some arguments are missing in the checkpoint.
    :param chkpt: a checkpoint.
    :param args: a argparse.Namespace. Sorry for the inconvenience.
    :return: a tuple (str, bool) with the dataset name and whether the model was trained by using the test set
             as a validation set (and thus, was trained with the full train set).
    """
    args_dataset = args_fun('dataset')
    args_use_test_set = args_fun('use_test_set_as_valid')
    if 'dataset' in chkpt:
        if args_dataset and chkpt['dataset']['name'] != args_dataset:
            print(f"Conflicting dataset for the model: {args_dataset} (args) "
                  f"VS {chkpt['dataset']['name']} (checkpoint)", file=sys.stderr)
            sys.exit(-2)
        dataset_name = chkpt['dataset']['name']
        use_test_set_as_valid = chkpt['dataset']['test_set_as_valid']
    else:
        dataset_name = args_dataset
        use_test_set_as_valid = args_use_test_set
    return dataset_name, use_test_set_as_valid


def get_num_classes(chkpt, args_fun):
    dataset_name, _ = load_dataset_args_from_checkpoint_or_args(chkpt, args_fun)
    num_classes = cifar.__dict__[dataset_name]('~/datasets', pin_memory=True).get_num_classes()
    return num_classes


def load_model_from_checkpoint_or_args(chkpt, args_fun, num_classes=None):
    """
    Loads a model from the given checkpoint, or from the args if some arguments are missing in the checkpoint.
    :param chkpt: a checkpoint.
    :param args_fun: a function that returns the appropriate values of the parameters.
                     This is so that we can use this function to load models and teachers,
                     even if their parameters have different names. NEEDED
    :param num_classes: number of classes of the dataset. Can be retrieved by loading a dataset again,
                        but can also be provided to save some time and memory.
    :return: a loaded model.
    """
    if num_classes is None:
        num_classes = get_num_classes(chkpt, args_fun)
    arch_name, base_width = load_model_args_from_checkpoint_or_args(chkpt, args_fun)
    model = models.__dict__[arch_name](
        num_classes=num_classes,
        base_width=base_width
    )
    model.cuda()
    model.load_state_dict(chkpt['state_dict'])
    return model


def load_teacher_from_checkpoint_or_args(args_fun, chkpt=None, num_classes=None):
    """
    Loads a teacher from the given checkpoint, or from the args if some arguments are missing in the checkpoint.
    Regarding the way this is implemented: here, chkpt would be the checkpoint of a student,
    that was trained with a teacher. So we get the path to the teacher from this checkpoint,
    then we load the checkpoint of the teacher, then we load the teacher as a regular model.

    :param args_fun: a function that returns appropriate values of the parameters.
    :param chkpt: a checkpoint. Passing None is helpful if you don't have a checkpoint and want to get values from args
    :param num_classes: number of classes.
    :return: a teacher, loaded appropriately, as well as the checkpoint that was used to load it.
    """
    if chkpt is None:
        chkpt = {}  # empty dictionary, that way no arg will be retrieved by it.
    if num_classes is None:
        num_classes = get_num_classes(chkpt, args_fun)
    teacher_path = args_fun('teacher_path')  # args override the teacher path; that way we can evaluate a student with another teacher
    if not teacher_path:
        # Okay, so we need to get it from the checkpoint
        if 'train_params' not in chkpt or 'distill' not in chkpt['train_params']:
            print(f"No teacher path available in the checkpoint; please provide on in the arguments; aborting")
            sys.exit(-2)
        d = chkpt['trains_params']['distill']
        teacher_path = d['teacher_path']
        teacher_path_rel = d['teacher_path_rel']
        teacher_path_abs = d['teacher_path_abs']
        print(f"""Info: {'''; '''.join(f"path {v} is {'valid' if os.path.isfile(v) else 'invalid'}"
                                       for k, v in d.items())}""", file=sys.stderr)
        if not os.path.isfile(teacher_path):
            # Okay, try relative path
            teacher_path = teacher_path_rel
        if not os.path.isfile(teacher_path):
            teacher_path = teacher_path_abs
        if not os.path.isfile(teacher_path):
            print(f"I'm sorry, I can't find a valid path for the teacher in the checkpoint; "
                  f"pass a correct path in the arguments; aborting", file=sys.stderr)
            sys.exit(-2)
    # Okay, so now we have a valid path name to a teacher; let's load it.
    chkpt_teacher = torch.load(teacher_path)
    teacher = load_model_from_checkpoint_or_args(chkpt_teacher, make_args_teacher_fun_from_fun(args_fun), num_classes)
    return teacher


def load_model_args_from_checkpoint_or_args(chkpt, args_fun):
    """
    Loads model arguments from the given checkpoint, or from the args if some arguments are missing in the checkpoint.
    Note that the 'args' argument is not a argparse.Namespace, but a function.
    This is so that we can use this function to load models and teachers, even if their parameters have different names
    in the checkpoint: we translate the names in this function.
    See :func:`make_args_fun` and :fund:`make_args_teacher_fun` for utility functions.

    :param chkpt: a checkpoint.
    :param args_fun: a function that returns the appropriate values of the parameters. NEEDED
    :return: a tuple (arch_name, base_width) of the model architecture parameters
    """
    args_arch = args_fun('arch')
    args_bw = args_fun('base_width')
    if 'arch' in chkpt:
        if args_arch and (chkpt['arch']['arch'] != args_arch or chkpt['arch']['base_width'] != args_bw):
            print(f"Conflicting architectures for the model: {args_arch}, "
                  f"{args_bw} (args) VS {chkpt['arch']} (checkpoint)", file=sys.stderr)
            sys.exit(-2)
        arch = chkpt['arch']['arch']
        base_width = chkpt['arch']['base_width']
    else:
        arch = args_arch
        base_width = args_bw
    return arch, base_width
