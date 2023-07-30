"""RDP utilities"""
import os
import sys
import logging
from typing import Iterable
import torch
from torch import nn
from scipy.optimize import bisect
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy \
    import compute_dp_sgd_privacy

log = logging.getLogger(__name__)


def compute_gaussian_sigma(
        epsilon, delta, batch_size, dataset_size, epochs) -> float:
    """Compute the level of noise to add when using rdp"""
    def compute_dp_sgd_wrapper(sigma):
        return compute_dp_sgd_privacy(
            n=dataset_size, batch_size=batch_size,
            noise_multiplier=sigma, epochs=epochs, delta=delta)[0] - epsilon

    # turn off output
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # calculate sigma
    sigma = bisect(compute_dp_sgd_wrapper, 1e-6, 1e6)
    # calculte actual privacy budget
    actual_epsilon = compute_dp_sgd_privacy(
        n=dataset_size, batch_size=batch_size, noise_multiplier=sigma,
        epochs=epochs, delta=delta)[0]
    log.info("Actual (ε,δ) is: ({}, {}), σ = {}".format(
        actual_epsilon, delta, sigma))

    # turn on output
    sys.stdout.close()
    sys.stdout = old_stdout
    return sigma


def clip_tensor(tensor: torch.Tensor, clip_bound):
    nn.utils.clip_grad_norm_(tensor, clip_bound)


def clip_gradients(net: torch.nn.Module, clip_bound):
    for param in net.parameters():
        clip_tensor(param, clip_bound)


def add_gaussian_noise(tensor: torch.Tensor, batch_size, sigma, clip_bound):
    """add noise to a list tensors"""
    noise_to_add = torch.zeros(
        tensor.shape, requires_grad=False).to(tensor.device)
    noise_to_add.normal_(0., std=clip_bound * sigma)
    noise = noise_to_add / float(batch_size)
    with torch.no_grad():
        tensor.add_(noise)
