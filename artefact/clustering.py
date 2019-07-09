#!/usr/bin/env python
import numpy as np
import pickle
from tqdm.autonotebook import tqdm

import torch
from torch import nn, from_numpy
from torch.autograd import Variable


from .autoencoder import Autoencoder
from .utils import kl_divergence, make_P

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class AutoencoderTSNE:
    def __init__(
        self,
        *,
        gpu=1,
        arch=None,
        learning_rate=0.001,
        weight_decay=0.01,
        lambda_kl=0.5,
        n_iter=5_000,
        algo_clustering=DBSCAN(eps=0.3, min_samples=7),
        filename_network=None,
        distance_trajectory='euclidean',
        verbose=True
    ):
        self.gpu = gpu
        self.arch = arch
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.lambda_kl = lambda_kl
        self.algo_clustering = algo_clustering
        self.n_iter = n_iter
        self.filename_network = filename_network
        self.distance_trajectory = distance_trajectory
        self.verbose = verbose

    def fit(self, X):
        P = torch.tensor(make_P(X, metric=self.distance_trajectory)).float().cuda(self.gpu)

        dim_input = X.shape[1]
        # if network's architecture not specified, use (n, n/2, 5)
        arch = self.arch if self.arch is not None else (dim_input, dim_input // 2, 2)

        model = Autoencoder(arch).cuda(self.gpu)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        d, k = [], []
        v = Variable(from_numpy(X.astype(np.float32))).cuda(self.gpu)
        for epoch in tqdm(range(self.n_iter)):

            lat = model.encoder(v)
            output = model.decoder(lat)

            dist = criterion(output, v)
            kl = kl_divergence(lat, P, self.gpu)
            loss =  dist + self.lambda_kl * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            d.append(dist.item())
            k.append(kl.item())


        if self.filename_network is not None:
            with open(self.filename_network, 'wb') as f:
                pickle.dump(model.cpu(), f)

        if self.verbose:
            plt.plot(d)
            plt.plot(k)

        lat = model.cpu().encoder(v.cpu()).detach().numpy()
        self.labels_ = self.algo_clustering.fit_predict(lat)
