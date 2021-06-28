import os
import shlex
import subprocess
import sys

lr_inits = ["5e-2", "1e-1"]
weight_decays = ["1e-4", "5e-4"]
widths = [32, 64]
depths = [20, 32, 44, 56, 34, 50]

width_idx_start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
depth_idx_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0

for width in widths[width_idx_start:]:
    for depth in depths[depth_idx_start:]:
        if depth * width**2 < 25000:
            continue
        teacher_name = f"resnet{depth}"
        for wd in weight_decays:
            for lr_init in lr_inits:
                cmd = shlex.split(f'python trainer.py --ds CIFAR100')
                cmd += ['-a', teacher_name, '--base-width', str(width)]
                cmd += ['--lr', lr_init, '--wd', wd]
                cmd += ['--log-dir', 'runs_teacher_2']
                print(f"Running process \"{' '.join(cmd)}\"")
                subprocess.run(
                    cmd
                )

