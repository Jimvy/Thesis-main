import subprocess
import sys

student_arch = 'resnet32'
student_width = 16
teacher_path = 'runs_teacher_2/CIFAR100_ResNet44-32_bs=128_lr=0.05_lr_dec=0.1_wd=0.0005_Jun08_02-18-28_gpu3/model.th'

wd = "1e-4"
lr_inits = ["0.1"]#, "0.2"]

#temps = [1, 2, 4]
#weights = ["1e-2", "1e-1", "3e-1", "1", "4"]
temps = [1, 2, 4, 6, 8]
weights = ["2e-2", "1e-1", "2e-1", "5e-1", "1", "2", "5"]

temp_idx_stat = int(sys.argv[1]) if len(sys.argv) > 1 else 0

for temp in temps[temp_idx_stat:]:
    for weight in weights:
        for lr_init in lr_inits:
            for _ in range(2):
                cmd = [
                    'python', 'trainer.py', '--ds', 'CIFAR100',
                    '-a', student_arch, '--base-width', str(student_width),
                    '--lr', str(lr_init), '--wd', wd,
                    '--distill', '--distill-weight', weight, '--distill-temp', str(temp),
                    '--teacher-path', teacher_path,
                    '--log-dir', 'runs_distill_4432_3216'
                ]
                print(f"Running process \"{' '.join(cmd)}\"")
                subprocess.run(cmd)

lr_inits = ["1e-1", "2e-1"]
for lr_init in lr_inits:
    for _ in range(3):
        cmd = [
            'python', 'trainer.py', '--ds', 'CIFAR100',
            '-a', student_arch, '--base-width', str(student_width),
            '--lr', str(lr_init), '--wd', wd,
            '--log-dir', 'runs_distill_4432_3216',
        ]
        print(f"Running process \"{' '.join(cmd)}\"")
        subprocess.run(cmd)

