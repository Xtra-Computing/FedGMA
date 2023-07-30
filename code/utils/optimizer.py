"""Create optimizers"""
import re
import copy
import inspect
from typing import Dict, Callable
import torch
from munch import Munch
from code.utils.arg_parser import TrainingConfParser, r_getattr


class OptimizerConfParser(TrainingConfParser):
    """Parser for optimizer config"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tags = self.argparse_tags if hasattr(self, 'argparse_tags') else ()
        stag, ltag = tuple(f"/{t}" for t in tags) if tags else ('', '')

        optimizer_args = self.add_argument_group(
            "Optimizer Configs (S,S->C)")

        optimizer_args.add_argument(
            f'-tr.o{stag}.n', f'--training.optimizer{ltag}.name',
            default='sgd', type=str, metavar='NAME',
            choices=optimizer_name_map.keys(),
            help="Optimizer for local training. \033[1mNOTE:\033[0m "
            "Detailed help messages for per-optimizer options are suppressed "
            "to avoid redundancy. The supported args are the same as pytorch's "
            "respective optimizer's initialization arguments (except for lr, "
            "which is initialized from --training.learning-rate). You can use "
            "--training.optimizer.<optimizer-name>.<arg-name> <arg-value> to "
            "specify them. E.g. you can specify "
            "--training.optimizer.sgd.momentum 0.9 to set a momentum of 0.9 "
            "for the sgd optimizer. or specify -tr.o.adam.bts 0.9 0.99 "
            "to specify a beta of (0.9, 0.99) for the adam optimizer")

        # Programmatically add arguments from torch.optim
        def shorten(key):
            return re.sub(r'(?<!^)[aeiou_]', '', key)

        for name, optim in optimizer_name_map.items():
            for key, param in inspect.signature(optim).parameters.items():
                if key in {'params', 'lr'} or key.endswith('_fn'):
                    continue

                # Fix option types
                type_args = {
                    'type': type(param.default[0]),
                    'nargs': len(param.default)
                } if isinstance(param.default, tuple) else {
                    'type': int
                } if any(pat in key for pat in ['max_eval']) else {
                    'type': bool
                } if key in ['foreach'] else {
                    'type': float
                } if any(pat in key for pat in ['momentum', 'decay', 'damp']) else {
                    'type': type(param.default)
                }
                assert type_args['type'] != type(type)
                assert type_args['type'] != type(None)

                optimizer_args.add_argument(
                    f"-tr.o{stag}.{shorten(name)}.{shorten(key)}",
                    f"--training.optimizer{ltag}.{name}.{key}", **type_args,
                    default=param.default, metavar=f"{key}".upper(),
                    help="%(type)s ")

            self.register_cfg_dep(
                f"training.optimizer{ltag}.{name}",
                lambda cfg, optim=name: r_getattr(
                    cfg, f"training.optimizer{ltag.replace('/', '.')}.name"
                ) == optim)


optimizer_name_map: Dict[str, Callable] = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'asgd': torch.optim.ASGD,
    'adamw': torch.optim.AdamW,
    'lbfgs': torch.optim.LBFGS,
    'nadam': torch.optim.NAdam,
    'radam': torch.optim.RAdam,
    'rprop': torch.optim.Rprop,
    'adamax': torch.optim.Adamax,
    'adagrad': torch.optim.Adagrad,
    'rmsprop': torch.optim.RMSprop,
    'adadelta': torch.optim.Adadelta,
    'sparseadam': torch.optim.SparseAdam,
}


def create_optimizer(cfg: Munch, params, tag: str = '', **kwargs):
    """
    Create an optimizer. parameters to optimize must be explicitly provied.
    For other arguments, if `kwargs` are provided, they will override
    corresponding options specified in `cfg`.

    By default, Learning rate is read from `cfg.training.learning_rate`, and
    other options are read from `cfg.training.optimizer.<name>`
    """
    cfg = copy.deepcopy(cfg)
    if tag:
        cfg.training.optimizer = cfg.training.optimizer[tag]
    return optimizer_name_map[cfg.training.optimizer.name](params, **{
        'lr': cfg.training.learning_rate,
        **cfg.training.optimizer[cfg.training.optimizer.name],
        **kwargs})
