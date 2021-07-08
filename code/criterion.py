import sys

import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    pass


class MultiCriterion(Criterion):
    def __init__(self):
        super().__init__()
        self.criterions_ot = []
        self.criterions_iot = []
        self.criterion_names = []
        self.prev_ret_map = {}

    def add_criterion(self, crit: Criterion, name: str, weight=1):
        if name in self.criterion_names:
            print("Warning, criterion with the same name {}".format(name), file=sys.stderr)
            id = 1
            new_name = "{}{}".format(name, id)
            while new_name in self.criterion_names:
                new_name = "{}{}".format(name, id)
                id += 1
            name = new_name
        self.criterion_names.append(name)
        self.criterions_iot.append((name, crit, weight))

    def forward(self, inputs, outputs, targets):
        self.prev_ret_map = self.get_criterions(inputs, outputs, targets)
        ret = 0
        for name, val in self.prev_ret_map.items():
            ret += val
        return ret

    def get_criterions(self, inputs, outputs, targets):
        ret = {}
        for name, crit, w in self.criterions_iot:
            ret[name] = w * crit(inputs, outputs, targets)
        return ret


class CrossEntropyLossCriterion(Criterion):
    def __init__(self):
        super().__init__()
        self._loss = (nn.CrossEntropyLoss().cuda(), )

    def forward(self, inputs, outputs, targets):
        return self._loss[0](outputs, targets)


class HKDCriterion(Criterion):
    def __init__(self, teacher, distill_temp, alpha=2):
        super().__init__()
        self._teacher = (teacher, )
        self._temp = distill_temp
        self._alpha = alpha

    def forward(self, inputs, outputs, targets):
        # softmaxfunc, logsoftmaxfunc, kldivfunc = nn.Softmax(dim=1).cuda(), nn.LogSoftmax(dim=1).cuda(), nn.KLDivLoss(reduction='batchmean').cuda()
        outputs_teacher = self._teacher[0](inputs).detach()
        # ret = (args.distill_temp ** 2) * kldivfunc(
        #     logsoftmaxfunc(outputs / args.distill_temp),
        #     softmaxfunc(outputs_teacher / args.distill_temp)
        # )
        ret = (self._temp ** self._alpha) * ((-1) * F.log_softmax(outputs / self._temp, dim=1) * F.softmax(outputs_teacher / self._temp, dim=1)).sum(dim=1).mean()
        return ret

