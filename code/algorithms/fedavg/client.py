"""simple FedAvg algorithm client"""
import sys
import time
from typing import Any, Dict
import torch
from code.compressors.none import NoCompress
from code.algorithms.base.base_client import BaseClient
from code.models import create_loss_criterion, create_model
from code.utils.metrics import create_metrics
from code.utils.optimizer import create_optimizer
from code.models.base import PerEpochTrainer
from code.communications.protocol import SyncInfos, MessageType, ClientInfo
from code.utils.training import all_params, seed_everything, init_buffer


class Client(BaseClient):
    """Simple client for the FedAvg algorithm"""

    def __init__(self, cfg):
        super().__init__(cfg)

        # the network to train on
        seed_everything(cfg.federation.seed)
        self.net = create_model(cfg, self.train_dataset)

        # the network to preserve the parameters/gradients of the last epoch,
        # for calculating the accumulated gradient with momentum
        # (not possible with only gradient accumulation on one network)
        self.gradient = create_model(cfg, dataset=self.train_dataset)
        # zero-initialization is essential for fedavg
        init_buffer(self.gradient, torch.device('cpu'))

        # ensure same global model initialization in client and server
        seed_everything(cfg.federation.seed)
        self.net_global = create_model(cfg, dataset=self.train_dataset)

        self.optimizer = create_optimizer(cfg, self.net.parameters())

        self.train_criterion = create_loss_criterion(cfg)
        self.additional_metrics = create_metrics(cfg)

        self.global_lr = cfg.training.learning_rate
        self.global_random_seed: int

        self.num_params = len(list(self.net_global.parameters()))

        self.iters = 0
        self.epochs = 0
        self.selected = False

        self.trainer = PerEpochTrainer(
            cfg, self.train_loader, self.train_criterion,
            self.additional_metrics, self.datalogger)

        # Storage for offloading to main storage
        if self.cfg.fedavg.advanced_memory_offload:
            self.offload_storage: Dict[str, Any] = {}

    def unload_resources(self):
        self.log.warning("Unloading resources")
        # Offload optimizer  state to main storage and delete the optimizer
        if self.cfg.fedavg.advanced_memory_offload:
            self.offload_storage['optim_state'] = self.optimizer.state_dict()[
                'state']
            for state in self.offload_storage['optim_state'].values():
                if state['momentum_buffer'] is not None:
                    state['momentum_buffer'] = state['momentum_buffer'].cpu()
            del self.optimizer
            for param in self.net.parameters():
                param.requires_grad_(False)
                param.grad = None

        self.net = self.net.cpu()
        self.gradient = self.gradient.cpu()
        self.net_global = self.net_global.cpu()
        self.train_criterion = self.train_criterion.cpu()
        super().unload_resources()

    def load_resources(self):
        super().load_resources()
        self.net = self.net.to(self.devices[0])
        self.gradient = self.gradient.to(self.devices[0])
        self.net_global = self.net_global.to(self.devices[0])
        self.train_criterion = self.train_criterion.to(self.devices[0])

        if self.cfg.fedavg.advanced_memory_offload:
            for param in self.net.parameters():
                param.requires_grad_(True)
            self.optimizer = create_optimizer(self.cfg, self.net.parameters())
            if 'optim_state' in self.offload_storage:
                for state in self.offload_storage['optim_state'].values():
                    if state['momentum_buffer'] is not None:
                        state['momentum_buffer'] = \
                            state['momentum_buffer'].to(self.devices[0])
                self.optimizer.state_dict()['state'] = \
                    self.offload_storage['optim_state']

    def train_one_round(self):
        """Train the model for one epoch/iter"""
        return self.trainer.train_model(self.net, self.optimizer, self.iters)

    def update_lr(self):
        """Update learning rate"""
        # TODO: support per-layer lr
        for group in self.optimizer.param_groups:
            group['lr'] = self.global_lr

    def handle_bye(self, msg_type):
        """Client exit if server exists"""
        if msg_type in {MessageType.BYE, MessageType._BYE}:
            self.log.info("Server gone. Stopping ...")
            sys.exit(0)

    def start_training(self, cfg):
        self.send_client_infos()

        # Send some metadata to aggregator
        self.optimizer.zero_grad()
        self.optimizer.step()
        self.log.info("Waiting for server's instruction ...")

        while True:
            # Receive sync information (also singal to start) from aggregator
            self.update_sync_info()

            # Shortcut if this client is not selected in this round
            if not self.selected:
                self.log.debug("Not updating this round")
                continue

            time_comm_round_start = time.time()
            self.init_comm_round()

            # Receive parameters from server and update global/local models
            self.update_global_params()

            # Acquire gpu must execute after network I/O, otherwise deadlock
            # (Acquired gpu resources but not able to receive server's
            # message since we cannot control the order of server broadcast)
            self.acquire_and_load_resources(cfg.hardware.gpus)

            # Update local parameters and learning rate
            self.update_local_params()
            self.update_lr()

            # do training
            self.train_model()
            self.test_model()

            self.calc_gradient()
            self.aggregate()

            time_comm_round_end = time.time()
            self.datalogger.add_scalar(
                "Time:CommRound", time_comm_round_end - time_comm_round_start,
                self.epochs)

            self.unload_resources_and_release(cfg.hardware.gpus)
            self.datalogger.flush()

    def send_client_infos(self):
        """Send client metadata to the aggregator"""
        self.dispatcher.send_msg(
            MessageType.CLIENT_INFOS, ClientInfo().set_data({
                'data_len': len(self.train_dataset),
            }).encode())

    def _log_evaluation_result(self, pfx, loss, results):
        self.log.info("  Test loss: %s", loss)
        for met, res in zip(self.additional_metrics, results):
            self.log.info("  Test %s: %s", met.name, res)

        self.log.debug("Writing to tensorboard")
        self.datalogger.add_scalar(f"{pfx}:loss", loss, self.epochs)
        for met, res in zip(self.additional_metrics, results):
            self.datalogger.add_scalar(f"{pfx}:{met.name}", res, self.epochs)

    def update_sync_info(self):
        """Update info in the synchronization message"""
        msg_type, data = self.dispatcher.recv_msg()
        self.handle_bye(msg_type)
        assert msg_type == MessageType.SYNC_INFOS

        sync_info = SyncInfos().decode(data)
        self._update_sync_info(sync_info)
        seed_everything(self.global_random_seed)    # sync global seed

    def _update_sync_info(self, sync_info: SyncInfos) -> None:
        self.global_lr = sync_info.data.lr
        self.global_random_seed = sync_info.data.seed
        self.selected = sync_info.data.selected

    def update_global_params(self):
        """Update the copy of global network"""
        msg_type, data = self.dispatcher.recv_msg()
        self.handle_bye(msg_type)
        params = NoCompress().decompress(data)
        with torch.no_grad():
            for g_param, param in zip(all_params(self.net_global), params):
                g_param.copy_(param.to(g_param.device))

    def update_local_params(self):
        """Update the gradient and the local network"""
        # update local gradient using global net, so new parameters could be
        # substracted from it later
        with torch.no_grad():
            for grad, param_g in zip(
                    all_params(self.gradient), all_params(self.net_global)):
                grad.copy_(param_g)

            for param, param_g in zip(
                    all_params(self.net), all_params(self.net_global)):
                param.copy_(param_g)

    def calc_gradient(self):
        """calculate gradient and store into self.gradient"""
        self.log.info("Calculating gradients...")

        # calculate gradient and store in `self.gradient`
        with torch.no_grad():
            for grad, param in zip(
                    all_params(self.gradient), all_params(self.net)):
                grad.copy_(grad - param)

    def aggregate(self):
        """Sent gradient info to aggregator"""
        data = NoCompress().compress(all_params(self.gradient))
        self.log.info("Sending gradient (%s bytes)", len(data))
        send_start = time.time()
        self.dispatcher.send_msg(MessageType.GRADIENT, data)
        send_end = time.time()
        self.datalogger.add_scalar(
            'Time:SendData', send_end - send_start, self.epochs)

        mem_stat = torch.cuda.memory_stats(self.devices[0])
        self.datalogger.add_scalar(
            "CUDA Mem Curr", mem_stat['allocated_bytes.all.current'], self.epochs)
        self.datalogger.add_scalar(
            "CUDA Mem Peak", mem_stat['allocated_bytes.all.peak'], self.epochs)

    def init_comm_round(self):
        """Initialize comm. round after receiving synchronization data"""
        self.log.info(
            "Client %s - Local Epoch: %s ==========",
            self.client_id, self.epochs + 1)
        self.log.info(
            "Learning rate: %s, seed: %s",
            self.optimizer.param_groups[0]['lr'], self.global_random_seed)

    def train_model(self):
        """Train the model"""
        time_training_start = time.time()
        self.net.train()
        for _ in range(self.cfg.fedavg.average_every):
            self.update_epoch_number()
            self.iters = self.train_one_round()
        time_training_end = time.time()
        self.log.info("Training finished")
        self.datalogger.add_scalar(
            "Time:Training", time_training_end - time_training_start,
            self.epochs)

    def update_epoch_number(self) -> None:
        """Update epoch number during training"""
        self.epochs += 1

    def test_model(self):
        """Test model before aggregate."""
        # (Test of the global model is done on the server)
        if self.epochs % self.cfg.federation.test_every == 0:
            self._test_model()

    def _test_model(self):
        for name, loader in self.test_loaders.items():
            test_loss, test_results = self.trainer.evaluate_model(
                self.net, loader, self.train_criterion,
                self.additional_metrics, self.devices[0])
            self._log_evaluation_result(
                f"Test({name})", test_loss, test_results)
        self.log.info("Testing finished")
