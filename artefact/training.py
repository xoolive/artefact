import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold._utils import _binary_search_perplexity
from scipy.spatial.distance import squareform
from tqdm.autonotebook import tqdm

"""
`_joint_probabilities` and `make_P` from the sklearn librairy:
https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/manifold/t_sne.py
"""
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
    assert np.all(P <= 1), "All probabilities should be less " "or then equal to one"
    return P


def kl_divergence(lat, P):
    dist = torch.nn.functional.pdist(lat, 2)
    dist = dist + 1.0
    dist = 1 / dist
    Q = dist / (torch.sum(dist))

    kl_divergence = 2.0 * torch.dot(P, P / Q)

    return kl_divergence


def train(
    model,
    X,
    device,
    nb_iterations,
    batch_size=1000,
    lambda_kl=0.05,
    distance_trajectory="euclidean",
    lr=1e-3,
    weight_decay=1e-5,
):

    model.to(device)
    model.train()
    with torch.enable_grad():
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        dataset = TensorDataset(torch.Tensor(X))
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            sampler=None,
            num_workers=10,
        )

        loss_evolution, re_evolution, kl_evolution = [], [], []

        for batch in tqdm(dataloader, desc="batches", leave=True, unit="batch"):
            v = batch[0].to(device)

            if lambda_kl > 0:
                P = torch.as_tensor(
                    make_P(batch[0], metric=distance_trajectory), device=device
                )

            for iteration in tqdm(range(nb_iterations), leave=False):
                lat, output = model(v)
                distance = nn.MSELoss(reduction="none")(output, v).sum(1).sqrt()

                loss = criterion(output, v)

                if lambda_kl > 0:
                    kl = kl_divergence(lat, P)
                    re_evolution.append(loss.cpu().item())
                    kl_evolution.append(kl.cpu().item())
                    loss += lambda_kl * kl

                loss_evolution.append(loss.cpu().item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return (
        model,
        {"loss": loss_evolution, "re_loss": re_evolution, "kl_loss": kl_evolution},
    )
