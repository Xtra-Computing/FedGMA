"""FL models"""
import copy
from itertools import chain
from typing import Callable, Dict, List, Optional, Type
from torch import nn
from munch import Munch
from torch.utils.data import DataLoader
from code.models.base import _ModelTrainer
from code.utils.metrics import _MetricCreator
from code.datasets.fl_split.base import _FLSplit
from code.utils.arg_parser import BaseConfParser, HWConfParser, r_getattr
from code.utils.logging import DataLogger
from code.utils.module import (
    camel_to_snake, load_all_direct_classes, load_all_submodules)

# Models ######################################################################

models_name_map: Dict[str, Type] = {
    camel_to_snake(name).replace('_', '-'): cls
    for module in load_all_submodules()
    for name, cls in load_all_direct_classes(module).items()
    if not name.startswith('_') and issubclass(cls, nn.Module)
}


class ModelConfParser(BaseConfParser):
    """Parse arguments for various machine learning models"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tags = self.argparse_tags if hasattr(self, 'argparse_tags') else ()
        stag, ltag = tuple(f"/{t}" for t in tags) if tags else ('', '')

        model_args = self.add_argument_group(
            "Model Initialization Configs (S,S->C)")

        model_args.add_argument(
            f'-md{stag}.n', f'--model{ltag}.name',
            default='resnet18', type=str, metavar='NAME',
            choices=models_name_map.keys(),
            help="Machine learning model to use. "
            f"Available models: {list(models_name_map.keys())}")

        model_args.add_argument(
            f'-md{stag}.l', f'--model{ltag}.loss',
            default='ce', type=str, metavar='NAME',
            choices=loss_criterion_name_map.keys(),
            help="Loss function to use. "
            f"Available functions: {list(loss_criterion_name_map.keys())}")

        for cls in chain.from_iterable(
                load_all_direct_classes(module).values()
                for module in load_all_submodules()):
            if not hasattr(cls, 'argparse_options'):
                continue
            cls.add_options_to(self, pfxs=('md', 'model'), tags=tags)

            for option in cls.argparse_options():
                self.register_cfg_dep(
                    f'model{ltag}.{option.flags[1]}',
                    lambda cfg, cls=cls: issubclass(models_name_map[r_getattr(
                        cfg, f"model{ltag.replace('/', '.')}.name")], cls))


def create_model(cfg: Munch, dataset: _FLSplit, tag: str = ''):
    """
    dataset: for reading dataset-dependent infos,
        e.g. number of classes, lstm embedding dim, etc.
    """
    cfg = copy.deepcopy(cfg)
    cfg.data = cfg.model[tag] if tag else cfg.model
    return models_name_map[cfg.model.name.lower()](cfg, dataset)


# Model Trainers ##############################################################

model_trainer_name_map = {
    camel_to_snake(name.removesuffix('Trainer')).replace('_', '-'): cls
    for module in load_all_submodules()
    for name, cls in load_all_direct_classes(module).items()
    if not name.startswith('_') and issubclass(cls, _ModelTrainer)
}


class ModelTrainerConfParser(HWConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tags = self.argparse_tags if hasattr(self, 'argparse_tag') else ()
        stag, ltag = (f'/{tags[0]}', f'/{tags[1]}') if tags else ('', '')

        model_trainer_args = self.add_argument_group("Model Trainer Configs")
        model_trainer_args.add_argument(
            f'-tr.mt{stag}.n', f'--training.model-trainer{ltag}.name',
            type=str, default='per-epoch', metavar='NAME',
            choices=model_trainer_name_map.keys(),
            help="Name of the model trainer. "
            f"Available: {list(model_trainer_name_map.keys())}")


def create_model_trainer(
        cfg: Munch, data_loader: DataLoader,
        loss_fn: Callable, metrics: List[_MetricCreator],
        datalogger: Optional[DataLogger] = None, tag: str = ''):
    """Create a model trainer"""

    cfg = copy.deepcopy(cfg)
    cfg.training.model_trainer = (
        cfg.training.model_trainer[tag] if tag else cfg.training.model_trainer)
    return model_trainer_name_map[cfg.training.model_trainer.name](
        cfg, data_loader, loss_fn, metrics, datalogger)


# Loss criterions #############################################################


loss_criterion_name_map: Dict[str, Type] = {
    'l1': nn.L1Loss,
    'mse': nn.MSELoss,
    'ce': nn.CrossEntropyLoss,
    'bce': nn.BCELoss,
}


def create_loss_criterion(cfg: Munch, tag: str = ''):
    """Crate loss calculator for backward propagation"""
    return (loss_criterion_name_map[cfg.model[tag].loss]() if tag else
            loss_criterion_name_map[cfg.model.loss]())
