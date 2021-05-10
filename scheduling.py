from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class LRSchedulerSequence(LRScheduler):
    def __init__(self, *args):
        self.schedulers = []
        for scheduler in args:
            if isinstance(scheduler, LRScheduler):
                self.schedulers.append(scheduler)

    def step(self, *args, **kwargs):
        for scheduler in self.schedulers:
            scheduler.step(*args, **kwargs)

    def add_scheduler(self, *args):
        for scheduler in args:
            self.schedulers.append(scheduler)

