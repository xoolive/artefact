import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.autonotebook import tqdm

from .utils import kl_divergence, make_P


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
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)

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
