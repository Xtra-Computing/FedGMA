"""Creating learning rate schedulers"""
import re
import copy
import inspect
import argparse
from typing import Dict, Callable
from munch import Munch
from torch.optim import lr_scheduler, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from code.utils.arg_parser import TrainingConfParser, r_getattr


class LRSchedulerConfParser(TrainingConfParser):
    """Parser for optimizer config"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tags = self.argparse_tags if hasattr(self, 'argparse_tags') else ()
        stag, ltag = tuple(f"/{t}" for t in tags) if tags else ('', '')

        scheduler_args = self.add_argument_group(
            "Learning Rate Scheduler Configs (S,S->C)")

        scheduler_args.add_argument(
            f'-tr.s{stag}.n', f'--training.scheduler{ltag}.name',
            default='reduceonplateau', type=str, metavar='NAME',
            choices=scheduler_name_map.keys(),
            help="Optimizer for local training. \033[1mNOTE:\033[0m "
            "Detailed help messages for per-scheduler options are suppressed "
            "to avoid redundancy. The supported args are the same as pytorch's "
            "respective scheduler's initialization arguments. You can use "
            "--training.scheduler.<optimizer-name>.<arg-name> <arg-value> to "
            "specify them. E.g. you can specify "
            "--training.scheduler.linear.total_iters 1000 to set iter of 1000 "
            "for the linear scheduler. or specify -tr.s.mltstp.gmm 0.2 "
            "to specify a gamma of 0.2 for the multistep scheduler.")

        # These are shceudlar args
        scheduler_args.add_argument(
            f'-tr.s{stag}.wd', f'--training.scheduler{ltag}.warmup-duration',
            default=0, type=int, metavar='N',
            help="Number of scheduler steps for warmup. If greater than 0, "
            "`create_scheduler` will wrap the scheduler specified by "
            "--training.scheduler.name into a linear warmup scheduler.")

        # Programmatically add arguments from torch.optim.lr_scheduler
        def shorten(key):
            return re.sub(r'(?<!^)[aeiou_]', '', key)

        for name, sched in scheduler_name_map.items():
            for key, param in inspect.signature(sched).parameters.items():
                # Skip certain keys
                if key == 'optimizer' or key.endswith('_fn'):
                    continue

                # Fix option types
                type_args = {
                    'type': int,
                    'nargs': '+'
                } if 'milestones' in key else {
                    'type': int,
                } if any(pat in key for pat in ['step', 'epoch', 'T_']) else {
                    'type': float,
                } if any(pat in key for pat in ['gamma', 'lambda', 'lr']) else {
                    'type': type(param.default)
                }

                assert type_args['type'] != type(type)
                assert type_args['type'] != type(None)

                scheduler_args.add_argument(
                    f"-tr.s{stag}.{shorten(name)}.{shorten(key)}",
                    f"--training.scheduler{ltag}.{name}.{key}",
                    **type_args, default=(
                        param.default if param.default != inspect._empty
                        else argparse.SUPPRESS
                    ), metavar=f"{key}".upper(),
                    help="%(type)s ")

            self.register_cfg_dep(
                f"training.scheduler{ltag}.{name}",
                lambda cfg, sched=name: r_getattr(
                    cfg, f"training.scheduler{ltag.replace('/', '.')}.name"
                ) == sched)


scheduler_name_map: Dict[str, Callable] = {
    'step': lr_scheduler.StepLR,
    'cyclic': lr_scheduler.CyclicLR,
    'linear': lr_scheduler.LinearLR,
    'constant': lr_scheduler.ConstantLR,
    'onecycle': lr_scheduler.OneCycleLR,
    'multistep': lr_scheduler.MultiStepLR,
    'exponential': lr_scheduler.ExponentialLR,
    'multiplicative': lr_scheduler.MultiplicativeLR,
    'cosineannealing': lr_scheduler.CosineAnnealingLR,
    'reduceonplateau': lr_scheduler.ReduceLROnPlateau,

    # The following schedulers are not included (they are composite schedulers)
    # 'chained': lr_scheduler.ChainedScheduler,
    # 'sequential': lr_scheduler.SequentialLR,
    # 'lambda': lr_scheduler.LambdaLR,
}


def create_scheduler(cfg: Munch, optimizer, tag: str = '', **kwargs):
    """
    Create a scheduler. optimizer to schedule must be explicitly provided.
    For other arguments, if `kwargs` are provided, they will override
    corresponding arguments specified in `cfg`.

    By default, options are read from `cfg.training.scheduler.<name>`
    """

    cfg = copy.deepcopy(cfg)
    if tag:
        cfg.training.scheduler = cfg.training.scheduler[tag]
    scheduler = scheduler_name_map[cfg.training.scheduler.name](optimizer, **{
        **cfg.training.scheduler[cfg.training.scheduler.name],
        **kwargs
    })

    warmup_duration = cfg.training.scheduler.warmup_duration
    assert warmup_duration >= 0, "Warmup duration must >= 0"

    return (GradualWarmupScheduler(
        optimizer, warmup_duration=cfg.training.scheduler.warmup_duration,
        after_scheduler=scheduler)
        if cfg.training.scheduler.warmup_duration > 0
        else scheduler)

    # Use SequentialLR also works but produces a deprecation warning:
    # return lr_scheduler.SequentialLR(
    #     optimizer, schedulers=[
    #         lr_scheduler.LinearLR(
    #             optimizer, start_factor=1./warmup_duration,
    #             end_factor=1., total_iters=warmup_duration),
    #         scheduler
    #     ], milestones=[warmup_duration]
    # ) if cfg.training.scheduler.warmup_duration > 0 else scheduler


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm-up(increasing) learning rate in optimizer.
    adapted from https://github.com/ildoonet/pytorch-gradual-warmup-lr
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_duration: duration for the learning rate to finish warmup
        after_scheduler: this scheduler is used after warmup finishs.
    """

    def __init__(self, optimizer: Optimizer, warmup_duration: int,
                 after_scheduler: _LRScheduler):
        self.warmup_duration = warmup_duration
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        """Overrides super().get_lr()"""
        if self.last_epoch > self.warmup_duration:
            if not self.finished:
                self.after_scheduler.base_lrs = copy.deepcopy(self.base_lrs)
                self.finished = True
            return self.after_scheduler.get_last_lr()

        return [base_lr * (float(self.last_epoch) / self.warmup_duration)
                for base_lr in self.base_lrs]

    def step_reduce_lr_on_plateau(self, metrics):
        """
        ReduceLROnPlateau is called at the end of epoch, whereas others are
        called at beginning
        """
        epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.warmup_duration:
            warmup_lr = [
                base_lr * (float(self.last_epoch) / self.warmup_duration)
                for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            self.after_scheduler.step(metrics)

    def step(self, epoch=None, metrics=None):
        if isinstance(self.after_scheduler, lr_scheduler.ReduceLROnPlateau):
            self.step_reduce_lr_on_plateau(metrics)
        else:
            if self.finished:
                self.after_scheduler.step(
                    epoch - self.warmup_duration if epoch else None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                super().step(epoch)
