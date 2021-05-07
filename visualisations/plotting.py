import argparse
from typing import Any, Dict, List, Tuple

import matplotlib.axes as mpaxes
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import torch
from torch import Tensor

from parsing import DISPLAYNAMES
from labelline import labelLines

__all__ = ['plot2d', 'plot3d', 'plot3dand2d']


def compute_lim(values: Tensor, min_proportion=0.95):
    r"""
    Computes "automatic" limits, that ensure that each line will have at least
    100*min_proportion % of its points drawn (ex: 90%).

    :param values: 2-D tensor of values that need to be plotted within some limits to determine.
                   Assumes a size of X x Y, where there are Y lines of X points each time.
    :param min_proportion: minimum proportion of points that need to be displayed for each curve.
    :return: min and max limits.
    """
    # 1st step: compute min and max
    dtype_f = torch.finfo(values.dtype)
    nX, nY = values.shape
    q = (1 - min_proportion) / 2  # proportion of points that can be outside the limits.
    nq = q * nX  # max number of points that can be outside of the limits
    # print(f"q = {q}, nq = {nq}")
    fixed_up_values_for_min = values.clone()
    fixed_up_values_for_min[torch.isnan(fixed_up_values_for_min)] = dtype_f.max
    fixed_up_values_for_min[torch.isinf(fixed_up_values_for_min)] = dtype_f.max
    fixed_up_values_for_max = values.clone()
    fixed_up_values_for_max[torch.isnan(fixed_up_values_for_max)] = dtype_f.min
    fixed_up_values_for_max[torch.isinf(fixed_up_values_for_max)] = dtype_f.min
    ymin = torch.min(fixed_up_values_for_min).data
    ymax = torch.max(fixed_up_values_for_max).data
    signif_diff = (ymax - ymin) / 100
    # print(f"  Determined values {ymin}, {ymax}")
    # 2nd step: binary search of the limit
    # First, min
    lim_min_l, lim_min_u = ymin, ymax
    # Below min_lim_l, I know that no point is present. Above the other, no point is present.
    while lim_min_u - lim_min_l > signif_diff:
        lim_min_m = (lim_min_u + lim_min_l) / 2
        is_ok = True  # Does it have at least 95% of points above it <=> at most 5% of points below it
        cnt_it = []
        for j in range(nY):
            val = torch.sum(fixed_up_values_for_min[:, j] < lim_min_m)
            cnt_it.append(val)
            if val > nq:
                is_ok = False
                # break
        # print(f"{lim_min_l}, {lim_min_u} ; {lim_min_m}; {cnt_it}")
        if is_ok:
            lim_min_l = lim_min_m
        else:
            lim_min_u = lim_min_m
    # print(f"  Post-binary search, determined limits min {lim_min_l}, {lim_min_u}")
    # Then, max
    lim_max_l, lim_max_u = ymin, ymax
    while lim_max_u - lim_max_l > signif_diff:
        lim_max_m = (lim_max_u + lim_max_l) / 2
        is_ok = True  # Does it have at least 95% of points above it <=> at most 5% of points below it
        cnt_it = []
        for j in range(nY):
            val = torch.sum(fixed_up_values_for_max[:, j] > lim_max_m)
            cnt_it.append(val)
            if val > nq:
                is_ok = False
                # break
        # print(f"{lim_max_l}, {lim_max_u} ; {lim_max_m}; {cnt_it}")
        if is_ok:
            lim_max_u = lim_max_m
        else:
            lim_max_l = lim_max_m
    # print(f"  Post-binary search, determined limits max {lim_max_l}, {lim_max_u}")
    delta_lims = (lim_max_u - lim_min_l)
    return lim_min_l - delta_lims / 100, lim_max_u + delta_lims / 100


def plot3d_graph(x: Tensor, y: Tensor, z: Tensor, /, axes: mpl_toolkits.mplot3d.Axes3D = None, *,
                 xlabel='x', xscale='linear', ylabel='y', yscale='linear', zlabel='loss', return_more=False):
    r"""
    3D plot.
    :param x: Should be a Tensor of size NxM, with rows repeated
    :param y: Should be a Tensor of size NxM too, with columns repeated
    :param z: Should be a Tensor of size NxM too
    :param axes: an already-created Axes object, or None if I should create a new one.
    :param xlabel: label of the first horizontal axis. Please be not too verbose
    :param xscale: scale of the first horizontal axis. Ex: 'log' or 'symlog'
    :param ylabel: label of the second horizontal axis. Please be not too verbose too.
    :param yscale: scale of the second horizontal axis. Ex: 'log' or 'symlog'
    :param zlabel: label of the vertical axis. Please be not too verbose too.
    :param return_more: Should I return more information that I have computed?
                        If True, I also return the (ylim_min, ylim_max) values.
    """
    if axes is None:
        axes = plt.gca()
        if isinstance(axes, mpl_toolkits.mplot3d.Axes3D):
            pass
        else:
            fig = plt.figure()
            axes = fig.add_subplot(projection='3d')
    # Okay, now we should have a correct axes.
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_zlabel(zlabel)
    zlim_min, zlim_max = compute_lim(z)
    surf = axes.plot_surface(x.numpy(), y.numpy(), z.numpy())
    if return_more:
        return surf, (zlim_min, zlim_max)
    else:
        return surf


def plot2d_graph(x: Tensor, y: Tensor, /, axes: mpaxes.Axes = None, labels=None, *,
                 xlabel='x', ylabel='loss', xscale='linear', yscale='linear', ylim=None, return_more=False):
    r"""
    Plots the specified (x, y) tensor, on a single Axes.
    It should be a detached Tensor (anyway, it's usefull for you too!).
    :param x: abscisses
    :param y: ordonnées
    :param axes: an already-created Axes object, or None if I should create a new one.
    :param labels: list of labels to display litteraly in-line.
    :param xlabel: label of the horizontal axis. Please be not too verbose
    :param xscale: scale of the horizontal axis. Ex: 'log' or 'symlog'
    :param ylabel: label of the vertical axis. Please be not too verbose too.
    :param yscale: scale of the vertical axis. Ex: 'log' or 'symlog'
    :param ylim: limits of the y axis. If not specified, limits are computed byt this plot.
    :param return_more: Should I return more information that I have computed?
                        If True, I also return the (ylim_min, ylim_max) values.
    :return:
    """
    if axes is None:
        axes = plt.gca()
    axes.tick_params(axis='both', which='both', labelsize='small')
    axes.set_xlabel(xlabel, fontsize='small', labelpad=0)
    axes.set_xscale(xscale)
    axes.set_ylabel(ylabel, fontsize='small', labelpad=0)
    axes.set_yscale(yscale)
    if ylim is None:
        ylim_min, ylim_max = compute_lim(y)
    else:
        ylim_min, ylim_max = ylim
    axes.set_ylim(ylim_min, ylim_max)
    lines = axes.plot(x.numpy(), y.numpy(), linewidth=1)
    axes.axhline(color='black', linestyle='-', linewidth=0.5)
    if labels:
        if len(labels) > 1:
            for line, label in zip(lines, labels):
                line.set_label(label)
            idx_mid = len(lines) // 2
            labelLines([lines[0], lines[-1]], size=7, outline_width=3)
        else:
            for line in lines:
                line.set_label(labels[0])
    leg = axes.legend(prop={'size': 7})
    leg.set_draggable(True)
    if return_more:
        return lines, (ylim_min, ylim_max)
    else:
        return lines


def plot_loss_comps_same_scale(x: Tensor, y: Tensor, zs: List[Tuple[Tensor, str]], labels=None, *,
                               fixed_vars_w_values='', vary_vars=None,
                               xlabel=None, xscale='linear', ylabel=None, yscale='linear', zlabel='loss', zscale='linear'):
    fig = plt.figure()
    fig.suptitle(f'Loss components, varying {vary_vars}, fixed {fixed_vars_w_values}')
    n = len(zs)
    zlim_min_all, zlim_max_all = 1e9, -1e9
    for z, _ in zs:
        zlim_min, zlim_max = compute_lim(z)
        zlim_min_all = min(zlim_min_all, zlim_min)
        zlim_max_all = max(zlim_max_all, zlim_max)
    for i, (zz, loss_name) in enumerate(zs):
        # ax3d = fig.add_subplot(2, n, i+1, projection='3d')  # type: mpl_toolkits.mplot3d.Axes3D
        # surf = plot3d_graph(x, y, zz, axes=ax3d,
        #                     xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale, zlabel=zlabel)
        # ax3d.title.set_text(f"Component {loss_name}")

        ax2d = fig.add_subplot(1, n, i+1)  # type: mpaxes.Axes
        lines = plot2d_graph(x, zz, axes=ax2d, labels=labels,
                             xlabel=xlabel, xscale=xscale, ylabel=zlabel, yscale=zscale,
                             ylim=(zlim_min_all, zlim_max_all), return_more=False)
        ax2d.title.set_text(f"Component {loss_name}")
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.12, top=0.85, wspace=0.2, hspace=0.1)


def plot_multi_grads_same_scale(x: Tensor, ys: List[Tuple[Tensor, str]], labels=None, *, xcomp=None,
                                fixed_vars_w_values='', vary_vars=None,
                                xlabel=None, ylabel='loss gradient', xscale='linear', yscale='linear'):
    fig = plt.figure()
    aaa = f'{xcomp[0]} and {xcomp[1]}'
    if vary_vars is None:
        vary_vars = aaa
    if len(ys) > 1:
        fig.suptitle(f'Gradient of full loss and of loss components, w.r.t. {aaa}, varying {vary_vars}, fixed {fixed_vars_w_values}')
    else:
        fig.suptitle(f'Gradient of {ys[0][1]}, varying {vary_vars}, fixed {fixed_vars_w_values}')
    if not xcomp:
        xcomp = [r'$a_S[1]$', r'$a_S[2]$']
    n = len(ys)
    ax2d_xs = []
    ax2d_ys = []
    ylim_min_all_x, ylim_max_all_x = 1e9, -1e9
    ylim_min_all_y, ylim_max_all_y = 1e9, -1e9
    for (y, _) in ys:
        ylim_min_x, ylim_max_x = compute_lim(y[:, :, 0])
        ylim_min_y, ylim_max_y = compute_lim(y[:, :, 1])
        ylim_min_all_x = min(ylim_min_all_x, ylim_min_x)
        ylim_max_all_x = max(ylim_max_all_x, ylim_max_x)
        ylim_min_all_y = min(ylim_min_all_y, ylim_min_y)
        ylim_max_all_y = max(ylim_max_all_y, ylim_max_y)
    lines_xs = []
    lines_ys = []
    for i, (y, y_name) in enumerate(ys):
        ax2d_x = fig.add_subplot(2, n, i+1)
        ax2d_x.title.set_text(f"Gradient of {y_name} w.r.t. {xcomp[0]}")
        ax2d_xs.append(ax2d_x)
        lines_x = plot2d_graph(x, y[:, :, 0], axes=ax2d_x, labels=labels,
                               xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale,
                               ylim=(ylim_min_all_x, ylim_max_all_x), return_more=False)

        ax2d_y = fig.add_subplot(2, n, n+i+1)
        ax2d_y.title.set_text(f"Gradient of {y_name} w.r.t. {xcomp[1]}")
        ax2d_ys.append(ax2d_y)
        lines_y = plot2d_graph(x, y[:, :, 1], axes=ax2d_y, labels=labels,
                               xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale,
                               ylim=(ylim_min_all_y, ylim_max_all_y), return_more=False)
        lines_xs.append(lines_x)
        lines_ys.append(lines_y)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.9, wspace=0.2, hspace=0.28)
    return lines_xs, lines_ys


def plot3dand2d(x: Tensor, y: Tensor, z: Tensor, args: argparse.Namespace, labels=None,
                vary_vars=None, fixed_vars_w_values=None):
    r"""
    3D + 2D plot, each in a subplot.
    See `plot3d` for the definition of the args.
    """
    fig = plt.figure()
    fig.suptitle(f'Full loss, varying {vary_vars}, fixed {fixed_vars_w_values}')
    xlabel = DISPLAYNAMES[args.vary_vars_list[0]['name']]
    xscale = args.vary_vars_list[0]['scale']
    ylabel = DISPLAYNAMES[args.vary_vars_list[1]['name']]
    yscale = args.vary_vars_list[1]['scale']
    zlabel = 'Loss'
    ax3d = fig.add_subplot(121, projection='3d')  # type: mpl_toolkits.mplot3d.Axes3D
    plot3d_graph(x, y, z, axes=ax3d,
                 xlabel=xlabel, xscale=xscale, ylabel=ylabel, yscale=yscale, zlabel=zlabel)

    ax2d = fig.add_subplot(122)  # type: mpaxes.Axes
    plot2d_graph(x, z, axes=ax2d, labels=labels, xlabel=xlabel, xscale=xscale, ylabel=zlabel)


