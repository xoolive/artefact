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
        gpu=None,
        model=0,
        learning_rate=0.001,
        weight_decay=0.01,
        lambda_kl=0.5,
        n_iter=5000,
        algo_clustering=DBSCAN(eps=0.3, min_samples=7),
        distance_trajectory="euclidean",
        verbose=True,
    ):
        # for now use the gpu 0 by default
        # TODO: handle the no gpu available case
        assert gpu is not None 
        self.gpu = gpu
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.lambda_kl = lambda_kl
        self.algo_clustering = algo_clustering
        self.n_iter = n_iter
        self.distance_trajectory = distance_trajectory
        self.verbose = verbose
        self.is_trained = False

    def fit(self, X):

        if not self.is_trained:
            reco_err, kls = self.train(X)

            if self.verbose:
                plt.figure(1)
                plt.subplot(121)
                plt.plot(reco_err)
                plt.title("reco err")

                plt.subplot(122)
                plt.plot(kls)
                plt.title("d_kl")

        v = Variable(from_numpy(X.astype(np.float32)))
        lat = self.model.encoder(v.cpu()).detach().numpy()
        self.labels_ = self.algo_clustering.fit_predict(lat)

    def to_pickle(self, filename):
        """Save the current AutoencoderTSNE in a pickle file named 'filename'
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, filename):
        """Load a file from pickle
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def score_samples(self, X):
        """Returns a numpy array containing the reconstruction error associated with each flight
        """
        v = Variable(from_numpy(X.astype(np.float32)))
        output = self.model(v)
        return nn.MSELoss(reduction="none")(output, v).sum(1).detach().numpy()

    def train(self, X):

        P = (
            torch.tensor(make_P(X, metric=self.distance_trajectory))
            .float()
            .cuda(self.gpu)
        )

        dim_input = X.shape[1]
        # if network's architecture not specified, use (n, n/2, 5)
        self.model = (
            self.model
            if self.model is not None
            else Autoencoder((dim_input, dim_input // 2, 2))
        )
        self.model.cuda(self.gpu)

        # dirty hack
        model_dim_input = next(self.model.parameters()).size()[1]
        assert model_dim_input == dim_input

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        mses, kls = [], []
        v = Variable(from_numpy(X.astype(np.float32))).cuda(self.gpu)
        for epoch in tqdm(range(self.n_iter)):

            lat = self.model.encoder(v)
            output = self.model.decoder(lat)

            dist = criterion(output, v)
            kl = kl_divergence(lat, P, self.gpu)
            loss = dist + self.lambda_kl * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mses.append(dist.item())
            kls.append(kl.item())

        # disable gpu after training
        self.model.cpu()

        self.is_trained = True
        return mses, kls
