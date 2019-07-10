#!/usr/bin/env python

"""
`_joint_probabilities` and `make_P` from the sklearn librairy:
https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/manifold/t_sne.py
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold._utils import _binary_search_perplexity
from scipy.spatial.distance import squareform

MACHINE_EPSILON_NP = np.finfo(np.float32).eps


def delta_max(x, y, *args):
    return np.max(np.abs(x - y))


def _joint_probabilities(distances, desired_perplexity, verbose=0):
    distances = distances.astype(np.float32, copy=True)
    conditional_P = _binary_search_perplexity(
        distances, None, desired_perplexity, verbose
    )
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON_NP)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON_NP)
    return P


def make_P(X, perplexity=30, metric="euclidiean"):
    distances = pairwise_distances(X, metric=metric)

    P = _joint_probabilities(distances, perplexity)
    assert np.all(np.isfinite(P)), "All probabilities should be finite"
    assert np.all(P >= 0), "All probabilities should be non-negative"
    assert np.all(P <= 1), (
        "All probabilities should be less " "or then equal to one"
    )
    return P


def kl_divergence(lat, P, gpu):
    dist = torch.nn.functional.pdist(lat.cpu(), 2).cuda(gpu)
    dist = dist + 1.0
    dist = 1 / dist
    Q = dist / (torch.sum(dist))

    kl_divergence = 2.0 * torch.dot(P, P / Q)

    return kl_divergence
