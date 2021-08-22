import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from functions import *
from parsing import parse_args, ALL_POSSIBLE_VARS, DISPLAYNAMES
from plotting import plot3dand2d, plot_multi_grads_same_scale, plot_loss_comps_same_scale
from utils_vizigoth import error, to_rect


def gen_samples(start: float, end: float, type='num', num=50, step=1, scale='lin', **kwargs) -> Tensor:
    if scale == 'log':
        start = np.log(start)
        end = np.log(end)
    if type == 'num':
        ret = torch.linspace(start, end, int(num))
    elif type == 'step':
        ret = torch.arange(start, end + step/10, step)
    else:
        ret = None
    if scale == 'log':
        ret = torch.exp(ret)
    return ret


def build_values(args: argparse.Namespace, var_names=ALL_POSSIBLE_VARS) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
    var_1_args = args.vary_vars_list[0]
    var_2_args = args.vary_vars_list[1]
    var_1_name = var_1_args['name']
    var_2_name = var_2_args['name']
    var_1_vals_1d = gen_samples(**var_1_args)  # size X
    var_2_vals_1d = gen_samples(**var_2_args)  # size Y
    var_1_vals_2d, var_2_vals_2d = torch.meshgrid([var_1_vals_1d, var_2_vals_1d])  # 2 x 2D grid of size X x Y
    var_1_vals_batched = var_1_vals_2d.reshape(-1).unsqueeze(dim=1)
    var_2_vals_batched = var_2_vals_2d.reshape(-1).unsqueeze(dim=1)
    B = var_1_vals_batched.shape[0]

    values = {}
    for var_name in var_names:
        x = getattr(args, var_name)
        if x is not None:
            values[var_name] = torch.tensor(x).repeat(B, 1)
        else:
            values[var_name] = x
    values[var_1_name] = var_1_vals_batched
    values[var_2_name] = var_2_vals_batched

    return values, var_1_vals_2d, var_2_vals_2d, var_1_vals_1d, var_2_vals_1d


def probs_from_acts(acts, t):
    return F.softmax(acts/t, dim=1)


def prepare_batch(values, infix='as', in_type='acts'):
    if in_type == 'acts':
        prefix = 'a' + infix + '_'
        acts_x = values[prefix + 'x']
        acts_y = values[prefix + 'y']
        acts_norm = values[prefix + 'norm']
        acts_angle = values[prefix + 'angle']
        x_batched, y_batched = to_rect(acts_x, acts_y, acts_norm, acts_angle)
    else:
        prefix = 'p' + infix + '_'
        if values[prefix + 'x'] is not None:
            probs_x = values[prefix + 'x']
            probs_y = 1 - probs_x
        else:
            assert values[prefix + 'y'] is not None
            probs_y = values[prefix + 'y']
            probs_x = 1 - probs_y
        x_batched, y_batched = probs_x, probs_y
    return x_batched, y_batched


def get_xcomp(in_type):
    if in_type == 'acts':
        return [DISPLAYNAMES['as_x'], DISPLAYNAMES['as_y']]
    else:
        return [DISPLAYNAMES['ps_x'], DISPLAYNAMES['ps_y']]


def get_fixed_vars(args):
    ret = []
    for var_name in ALL_POSSIBLE_VARS:
        x = getattr(args, var_name)
        if x is not None:
            ret.append(var_name)
    return ret


def get_vary_vars(args):
    return [ll['name'] for ll in args.vary_vars_list]


def main(args):
    fixed_vars = get_fixed_vars(args)
    vary_vars = get_vary_vars(args)
    values, first_var_2d, second_var_2d, first_var_1d, second_var_1d = build_values(args)
    print(second_var_1d)
    tau_batched = values['temp']
    s_x_batched, s_y_batched = prepare_batch(values, infix='s', in_type=args.student_input_type)
    t_x_batched, t_y_batched = prepare_batch(values, infix='t', in_type=args.teacher_input_type)

    ins_student_batched = torch.cat((s_x_batched, s_y_batched), dim=1)
    # ins_student_batched.requires_grad_(True)
    ins_teacher_batched = torch.cat((t_x_batched, t_y_batched), dim=1)

    Y = torch.zeros(ins_student_batched.shape[0], 1)
    grad_Y = torch.zeros(ins_student_batched.shape[0], 2)
    partial_Y = []
    partial_grads = []
    print("Y", Y.shape)
    loss_names = []
    for loss_name, weight, more_ops in args.losses_list:
        if loss_name == 'HKD':
            if len(more_ops) > 0:
                _d = {'1': 1, '2': 2, '0': 0}
                if more_ops[0] not in _d:
                    error(f"Invalid power \'{more_ops[0]}\'for tau for loss HKD")
                tau_pow = _d[more_ops[0]]
                if len(more_ops) > 1:
                    error(f"Unkown additional options for HKD loss: {more_ops[1:]}")
            else:
                tau_pow = 2
            L = weight * hkd_loss(s_ins=ins_student_batched, t_ins=ins_teacher_batched, t=tau_batched,
                                  s_in_type=args.student_input_type, t_in_type=args.teacher_input_type, tau_pow=tau_pow)
            # input_grad = torch.ones(ins_student_batched.shape[0], 1)
            # L.backward(input_grad)
            # torch_grad_L = ins_student_batched.grad
            grad_L = weight * hkd_loss_grad(ins_student_batched, ins_teacher_batched, tau_batched,
                                            s_in_type=args.student_input_type, t_in_type=args.teacher_input_type, tau_pow=tau_pow)
        elif loss_name == 'CE':
            L = weight * ce_loss(ins_student_batched, in_type=args.student_input_type)
            grad_L = weight * ce_loss_grad(ins_student_batched, in_type=args.student_input_type)
        else:
            error(f"Unknown loss \'{loss_name}\', bug in program")
            L = 0  # PyCharm is annoying
            grad_L = 0
        Y += L
        grad_Y += grad_L
        loss_names.append(f"{loss_name} loss weight {weight}")
        partial_Y.append((L, loss_names[-1]))
        partial_grads.append((grad_L, loss_names[-1]))
        # partial_grads.append((torch_grad_L, loss_names[-1] + ' autograd'))
    Y_unflattened = torch.reshape(Y, first_var_2d.shape)
    print(Y_unflattened.shape)
    partial_Y_unflattened = [(torch.reshape(YY.detach(), first_var_2d.shape), YYY) for YY, YYY in partial_Y]
    grad_Y_unflattened = torch.reshape(grad_Y.detach(), (first_var_2d.shape[0], first_var_2d.shape[1], 2))
    partial_grads_unflattened = [(torch.reshape(g.detach(), (first_var_2d.shape[0], first_var_2d.shape[1], 2)), l) for g, l in partial_grads]
    fixed_vars_w_values = ', '.join(f'{DISPLAYNAMES[var_name]}={getattr(args, var_name)}' for var_name in fixed_vars)
    second_var_labels = [f"{DISPLAYNAMES[args.vary_vars_list[1]['name']]}={i.item():3.2}" for i in second_var_1d]
    vary_vars_s = ', '.join(DISPLAYNAMES[v] for v in vary_vars)

    plot3dand2d(first_var_2d, second_var_2d, Y_unflattened, args, labels=second_var_labels,
                vary_vars=vary_vars_s, fixed_vars_w_values=fixed_vars_w_values)

    # Plot the loss components
    if len(partial_Y_unflattened) > 1:
        ys = [(Y_unflattened, "full loss")] + partial_Y_unflattened
    else:
        ys = partial_Y_unflattened
    plot_loss_comps_same_scale(first_var_2d.detach(), second_var_2d.detach(), ys,
                               labels=second_var_labels,
                               fixed_vars_w_values=fixed_vars_w_values, vary_vars=vary_vars_s,
                               xlabel=DISPLAYNAMES[args.vary_vars_list[0]['name']],
                               ylabel=DISPLAYNAMES[args.vary_vars_list[1]['name']],
                               xscale=args.vary_vars_list[0]['scale'],
                               yscale=args.vary_vars_list[1]['scale'])

    # Plot the gradients
    if len(partial_grads_unflattened) > 1:
        ys = [(grad_Y_unflattened, "full loss")] + partial_grads_unflattened
    else:
        ys = partial_grads_unflattened
    plot_multi_grads_same_scale(first_var_2d.detach(), ys, labels=second_var_labels,
                                fixed_vars_w_values=fixed_vars_w_values, vary_vars=vary_vars_s,
                                xcomp=get_xcomp(args.student_input_type),
                                xlabel=DISPLAYNAMES[args.vary_vars_list[0]['name']],
                                xscale=args.vary_vars_list[0]['scale'])
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
