"""A list of metrics for training"""
import re
import copy
import inspect
import argparse
from typing import List, Callable, Union, get_args, get_origin, get_type_hints
import torchmetrics
from munch import Munch
from code.utils.arg_parser import TrainingConfParser, r_getattr


class MetricsConfParser(TrainingConfParser):
    """Parser for optimizer config"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tags = self.argparse_tags if hasattr(self, 'argparse_tags') else ()
        stag, ltag = tuple(f"/{t}" for t in tags) if tags else ('', '')

        optimizer_args = self.add_argument_group(
            "Metrics Configs (S,S->C)")

        optimizer_args.add_argument(
            f'-tr.m{stag}.ns', f'--training.metric{ltag}.names',
            default=['accuracy'], type=str, nargs='+', metavar='NAME',
            choices=metrics_name_map.keys(),
            help="Metrics for both server and clients. Available metrics: "
            f"{list(metrics_name_map.keys())}")

        # Programmatically add arguments from torch.optim
        def shorten(key):
            return re.sub(r'(?<!^)[aeiou_]', '', key)

        for name, optim in metrics_name_map.items():
            for key, param in inspect.signature(optim).parameters.items():
                if key in {'kwargs'}:
                    continue

                # Fix option types with type hints
                type_args = {
                    'type': get_args(get_type_hints(optim.__init__)[key])[0]
                } if (param.default is None and
                      key in get_type_hints(optim.__init__) and
                      get_origin(get_type_hints(optim.__init__)[key]) is Union) else {
                    'type': str,
                    'choices': ['binary', 'multiclass', 'multilabel'],
                } if optim == torchmetrics.F1Score and key == 'task' else {
                    'type': int,
                } if key == 'num_labels' else {
                    'type': type(param.default),
                }

                assert type_args['type'] != type(type)
                assert type_args['type'] != type(None)

                optimizer_args.add_argument(
                    f"-tr.m{stag}.{shorten(name)}.{shorten(key)}",
                    f"--training.metric{ltag}.{name}.{key}", **type_args,
                    default=(
                        param.default if param.default != inspect._empty else
                        argparse.SUPPRESS
                    ), metavar=f"{key}".upper(),
                    help="%(type)s ")

            self.register_cfg_dep(
                f"training.metric{ltag}.{name}",
                lambda cfg, name=name: name in r_getattr(
                    cfg, f"training.metric{ltag.replace('/','.')}.names"))


metrics_name_map = {
    'accuracy': torchmetrics.Accuracy,
    'auc': torchmetrics.AUC,
    'mse': torchmetrics.MeanSquaredError,
    'mae': torchmetrics.MeanAbsoluteError,
    'f1': torchmetrics.F1Score,
    'ppl': torchmetrics.Perplexity,
    # To add metrics, ensure that the added classs have `update` and `compute`
    # methods. The `update` methods should take (pred, label) as args, while
    # the `compute` methods take no additional args other than `self`
}


class _MetricCreator:
    def __init__(self, name, **kwargs):
        assert name in metrics_name_map
        assert hasattr(metrics_name_map[name], 'update')
        assert hasattr(metrics_name_map[name], 'compute')

        self.name = name
        self.metric_class = metrics_name_map[name]
        self.kwargs = kwargs

    def __call__(self):
        return self.metric_class(**self.kwargs)


def create_metrics(cfg: Munch, tag: str = '', **kwargs) -> List[_MetricCreator]:
    """
    Create list of metrics creators. If `kwargs` are provided,
    they will override corresponding options specified in `cfg`.

    Creators instead metroc objects are returned to avoid re-using the same
    metric object. Re-using the same metric object may lead to false evaluation.
    """

    cfg = copy.deepcopy(cfg)
    if tag:
        cfg.training.metric = cfg.training.metric[tag]
    return [_MetricCreator(name, **{**cfg.training.metric[name], **kwargs})
            for name in cfg.training.metric.names]
