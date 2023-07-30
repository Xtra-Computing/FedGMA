"""Base class of all FL servers for the C/S architecture"""
import os
import json
import signal
import logging
import threading
from functools import partial
from abc import ABC, abstractmethod
from typing import Callable, Dict, List
from munch import Munch
import torch
from torch.utils.data import DataLoader
from code.algorithms.base.base import Node
from code.utils.scheduler import LRSchedulerConfParser
from code.utils.training import init_determinism
from code.utils.logging import LogConfParser, init_logging, DataLogger
from code.utils.optimizer import OptimizerConfParser
from code.datasets import DataConfParser, create_dataset
from code.datasets.raw_dataset import dataset_name_to_class
from code.models import ModelConfParser
from code.utils.metrics import MetricsConfParser
from code.utils.arg_parser import (
    FLConfParser, HWConfParser, TrainingConfParser)
from code.communications.dispatcher import (
    ServerDispatcherConfParser, ServerDispatcher)
from code.communications.protocol import (
    Protocol, MessageType, ForwardedConfig)


class BaseServerConfParser(
        LogConfParser, HWConfParser, ModelConfParser, DataConfParser,
        MetricsConfParser, OptimizerConfParser, LRSchedulerConfParser,
        TrainingConfParser, ServerDispatcherConfParser, FLConfParser):
    """Parse base configs that will used by the server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cs_args = self.add_argument_group(
            "Client/Sever Architecture Configs (S,S->C)")
        cs_args.add_argument(
            '-cs.n', '--client-server.num-clients',
            default=4, type=int, required=True, metavar='N',
            help="Number of clients")

        cs_args.add_argument(
            '-cs.td', '--client-server.test-datasets',
            nargs='+', metavar="DATASET",
            choices=dataset_name_to_class.keys(),
            help="Seleted test datasets must have the same prefix, "
            "Combined test dataset can be separted with ',' (without space), "
            "e.g. 'amazon,imdb'.\n"
            "Will test on all datasets if multiple datasets are specified.\n"
            "Will use the test set tied to the train set if set to None.")

        server_args = self.add_argument_group(
            "Server-Only Configs (S)")

        server_args.add_argument(
            '-sv.nc', '--server.num-cache',
            default=5, type=int, metavar='N',
            help="# of received client models that exist at the same time\n"
            " in memory, 2 * nc should be less than total host memory size")

    @staticmethod
    def filter_server_cfg(cfg: dict) -> Munch:
        """Remove all arguments under the namesapce `server.`"""
        # Filter out server-specific configs
        return Munch.fromDict({
            k: v for (k, v) in cfg.items()
            if k not in {'server', 'log', 'hardware', 'dispatcher'}})


class BaseServerScheduler:
    """Scheduler for the server. Records client info and handles basic events"""

    def __init__(self, cfg: Munch, datalogger=None) -> None:
        self.log = logging.getLogger(self.__class__.__name__)

        self.dispatcher = ServerDispatcher(cfg, datalogger=datalogger)
        self.start = self.dispatcher.start

        self.cfg_for_client = BaseServerConfParser.filter_server_cfg(cfg)
        """Configs to send to the client upon receiving a handshake"""

        self.clients_socket_id_map: Dict[int, int] = {}
        """Map socket to client_id"""
        self.clients_id_socket_map: Dict[int, int] = {}
        """Map client_id to socket"""

        # the number of clients connected
        self.num_clients_connected = 0
        """Total number of clients connected"""
        self.clients_info_lock = threading.Lock()
        """Lock for updating client info"""

        self.event_all_clients_connected = threading.Event()
        """A One-shot event signaling the specified num of clients connected"""

        self.cleanup_hooks: List[Callable[[], None]] = []
        """A list of cleanup hooks to execute at cleanup"""

        num_clients = cfg.client_server.num_clients
        self.dispatcher.register_msg_event(
            MessageType.HANDSHAKE,
            partial(self.process_handshake, num_clients=num_clients))
        self.dispatcher.register_msg_event(
            MessageType._BYE, self.process_bye)
        self.dispatcher.register_shutdown_event(
            self.process_dispatcher_shutdown)

        self.init_signal_handler()

    def process_handshake(self, socket, data, /, num_clients):
        """Verify handshake message and assign client id"""

        if data != b'Hello!':
            self.log.warning("Handshake message verification failed!")
            return

        with self.clients_info_lock:
            # Generate config to send to client, assign client id
            msg = ForwardedConfig(self.cfg_for_client)
            msg.data['client_id'] = self.num_clients_connected
            self.clients_socket_id_map[socket] = msg.data.client_id
            self.clients_id_socket_map[msg.data.client_id] = socket
            self.log.info(
                "Client %s --> fd=%s ✔", msg.data.client_id, socket.fileno())

            self.dispatcher.schedule_task(
                Protocol.send_data, socket,
                MessageType.TRAINING_CONFIG, msg.encode())

            # if number of clients is enough, start training
            self.num_clients_connected += 1
            if self.num_clients_connected == num_clients:
                self.event_all_clients_connected.set()

    def process_bye(self, socket, _):
        """
        Process `BYE` message from client
        This determins how and when should the sever stop in a normal path.
        """
        # skip non-registered clients
        if socket not in self.clients_socket_id_map:
            return

        with self.clients_info_lock:
            client_id = self.clients_socket_id_map[socket]
            self.log.info("Connection to client %s closed (fd=%s)",
                          client_id, socket.fileno())
            # self.clients_socket_id_map.pop(socket)
            # self.clients_id_socket_map.pop(client_id)

            self.num_clients_connected -= 1

            # if all clients leaves while training already started, it means
            # a abnormal state. force shutdown. (normal shutdown should be
            # handled by subclass protocol)
            if self.event_all_clients_connected.is_set() and \
                    self.num_clients_connected == 0:
                self.log.info("All clients are gone, shutting down...")
                self.cleanup()
                # this function is called from dispatcher thread, so force
                # stopping the main thread is necessary. see:
                # https://stackoverflow.com/questions/1489669
                os._exit(1)

    def process_dispatcher_shutdown(self):
        """Handles dispatcher abnormal shutdown"""
        self.log.warning("Exception in server. Stopping ...")
        self.cleanup()
        os._exit(1)

    def cleanup(self):
        """common cleanup hooks"""
        # Ignore exception at cleanup stage
        try:
            for hook in self.cleanup_hooks:
                hook()
        except Exception as exc:  # pylint: disable=broad-except
            self.log.exception(
                "Exception exceuting cleanup hook", exc_info=exc)
        finally:
            # cleanup and close all sockets gracefully,
            # in both exception-path and normal-path
            self.dispatcher.stop()

    def register_cleanup_hook(self, hook: Callable[[], None]) -> None:
        """
        Register a hook to be called when cleanup. Note: hook might be called
        from any thread (main/dispatcher/task scheduler).
        """
        self.cleanup_hooks.append(hook)

    def init_signal_handler(self):
        """Initializes handler for the SIGINT (Ctrl-C) signal"""
        signal.signal(signal.SIGINT, self.sigint_handler)

    def sigint_handler(self, *_, **__):
        """Handler for the SIGINT singal"""
        self.log.warning('SIGINT detected. Stopping everything...')
        self.cleanup()
        os._exit(0)


class BaseSever(Node, ABC):
    """Base server for federated learning"""
    @classmethod
    def conf_parser(cls):
        return BaseServerConfParser

    def __init__(self, cfg, scheduler=BaseServerScheduler):
        super().__init__()

        # Number of clients must match number of data splits
        assert cfg.client_server.num_clients == cfg.data.fl_split.num

        # Store config
        self.cfg = cfg

        # Initialize CUDA context on the first listed GPU
        if len(cfg.hardware.gpus) > 0:
            torch.cuda.device(cfg.hardware.gpus[0])

        # Note down devices to use
        self.devices = ([
            torch.device('cuda', gpu_idx) for gpu_idx in cfg.hardware.gpus]
            if len(cfg.hardware.gpus) > 0 else [torch.device('cpu')])

        # Initialize infrastructures
        init_logging(cfg)
        init_determinism(cfg)

        # Initialize logging and tensorboard writer
        self.log = logging.getLogger(self.__class__.__name__)
        self.datalogger = DataLogger(cfg, 'Server')
        self.log.info("\n%s", json.dumps(cfg, indent=2))

        # Avoid passing garbage to clients
        self.__init_dataset(cfg)
        self.scheduler = scheduler(cfg)

    @abstractmethod
    def start_training(self, cfg):
        """Subclass should override this method to define the training alg."""
        self.scheduler.register_cleanup_hook(self.cleanup_hook)
        self.scheduler.start()

    def __init_dataset(self, cfg: Munch):
        """Initialize test and eval dataset for the server"""
        self.eval_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='eval', split_id=None)

        test_datasets = (
            cfg.client_server.test_datasets
            if cfg.client_server.test_datasets
            else [','.join(cfg.data.raw.datasets)])

        self.test_datasets = {
            tdss: create_dataset(
                cfg, datasets=tdss.split(','), mode='test', split_id=None)
            for tdss in test_datasets
        }

        eval_g = torch.Generator()
        eval_g.manual_seed(cfg.federation.seed)
        self.eval_loader = DataLoader(
            self.eval_dataset, cfg.training.batch_size,
            shuffle=False, generator=eval_g)

        test_gs = {k: torch.Generator() for k in self.test_datasets.keys()}
        for generator in test_gs.values():
            generator.manual_seed(cfg.federation.seed)

        self.test_loaders = {
            k: DataLoader(
                v, cfg.training.batch_size,
                shuffle=False, generator=test_gs[k])
            for k, v in self.test_datasets.items()
        }

    def cleanup(self):
        self.scheduler.cleanup()

    def cleanup_hook(self):
        """Cleanup function. Called at dispatcher shutdown"""
        self.datalogger.close()
