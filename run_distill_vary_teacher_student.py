import subprocess
import sys

wt_idx_start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stud_idx_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
teach_idx_start = int(sys.argv[3]) if len(sys.argv) > 3 else 0

weight_temp_confs = [
    (0.5, 4),
    # (0.5, 6),
    # (1, 6),
    (2, 6),
]

teachers = [
    #   models with high WD and fully trained
    'teachers/CIFAR100_ResNet44-64/model_best.th',  # 78.3
    'teachers/CIFAR100_ResNet20-64/model_best.th',  # 77.6
    'teachers/CIFAR100_ResNet44-32/model_best.th',  # 75.4
    #   models with low WD and fully trained
    'teachers/CIFAR100_ResNet44-64/model_best_smallwd.th',  # 75.1
    # 'teachers/CIFAR100_ResNet20-64/model_best_smallwd.th',  # 74.6
    'teachers/CIFAR100_ResNet32-32/model_best_smallwd.th',  # 72.4, RN44-32 was bad
    #   models trained before epoch 120, so-called "early stopped"
    # 'teachers/CIFAR100_ResNet32-64/model_best_early.th',  # 77.5
    'teachers/CIFAR100_ResNet20-64/model_best_early.th',  # 76.3
    'teachers/CIFAR100_ResNet32-32/model_best_early.th',  # 74.3
    'teachers/CIFAR100_ResNet20-64/model_best_smallwd_early.th',  # 74.4
]

student_confs = [
    (20, 16),
    (20, 24),
    (20, 32),
    (32, 16),
    (32, 24),
    (44, 8),
    (44, 16),
    (44, 24),
    (56, 8),
    (56, 16),
    # (34, 16),
    # (50, 16),
]

lr_inits = ["0.1"]
wd = "1e-4"

for student_depth, student_width in student_confs[stud_idx_start:]:
    student_arch = f"resnet{student_depth}"
    for teacher in teachers[teach_idx_start:]:
        # parameter variation, to check
        for lr_init in lr_inits:
            for weight, temp in weight_temp_confs[wt_idx_start:]:
                for _ in range(2):
                    cmd = [
                        'python', 'trainer.py', '--ds', 'CIFAR100',
                        '-a', student_arch, '--base-width', str(student_width),
                        '--lr', str(lr_init), '--wd', wd,
                        '--distill', '--distill-weight', str(weight),
                        '--distill-temp', str(temp),
                        '--teacher-path', teacher,
                        '--log-dir', 'runs_distill_multi_teach_studs',
                    ]
                    print(f"Running process \"{' '.join(cmd)}\"")
                    subprocess.run(cmd)
    # No distillation training
    for lr_init in lr_inits:
        for _ in range(1):
            cmd = [
                    'python', 'trainer.py', '--ds', 'CIFAR100',
                    '-a', student_arch, '--base-width', str(student_width),
                    '--lr', str(lr_init), '--wd', wd,
                    '--log-dir', 'runs_distill_multi_teach_studs',
                ]
            subprocess.run(cmd)

