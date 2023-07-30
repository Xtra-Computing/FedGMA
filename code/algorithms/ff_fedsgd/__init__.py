"""The FedSGD algorithms"""
from code.algorithms.fedsgd import _FedSGDAdaptor
from code.algorithms import ff_fedavg


class Server(_FedSGDAdaptor, ff_fedavg.Server):
    """Full-feature FedSGD server"""


class Client(ff_fedavg.Client):
    """FedSGD client is the same as fedavg client"""
