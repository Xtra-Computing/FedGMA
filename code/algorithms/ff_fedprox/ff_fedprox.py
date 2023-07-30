from code.algorithms.base.ff_base_server import FFBaseServerConfParser
from code.algorithms import ff_fedavg
from code.models.base import PerEpochTrainer, PerIterTrainer
from code.algorithms import fedprox


class ConfParser(fedprox.ConfParser, FFBaseServerConfParser):
    """Full-feature FedProx config parser"""


class Server(ff_fedavg.Server):
    @classmethod
    def conf_parser(cls):
        return ConfParser


class Client(fedprox.Client, ff_fedavg.Client):
    """Full feature FedProx algorithm's client """

    def __init__(self, cfg):
        super().__init__(cfg)

        if isinstance(self.trainer, PerEpochTrainer):
            self.trainer = fedprox.PerEpochTrainer(
                cfg, self.train_loader, self.train_criterion,
                self.additional_metrics, self.datalogger)
        if isinstance(self.trainer, PerIterTrainer):
            self.trainer = fedprox.PerIterTrainer(
                cfg, self.train_loader, self.train_criterion,
                self.additional_metrics, self.datalogger)
        else:
            raise RuntimeError("Model trainer not initailized")
