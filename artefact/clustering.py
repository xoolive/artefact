#!/usr/bin/env python
import numpy as np
from random import sample
import pickle
from tqdm.autonotebook import tqdm

import torch
from torch import nn, from_numpy
from torch.autograd import Variable


from .autoencoder import Autoencoder
from .utils import kl_divergence, make_P

from sklearn.cluster import DBSCAN


class AutoencoderTSNE:
    def __init__(
        self,
        *,
        gpu=0,
        model=None,
        learning_rate=0.001,
        weight_decay=0.01,
        lambda_kl=0.05,
        n_iter=5000,
        algo_clustering=DBSCAN(eps=0.3, min_samples=7),
        batch_size=500,
        nb_batches=10,
        distance_trajectory="euclidean",
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
        self.is_trained = False
        self.reco_err, self.kls = [], []
        self.batch_size = batch_size
        self.nb_batches = nb_batches
        self.X = None
        self.hash_X = None

    def fit(self, X):
        h_X = hash(X.tobytes())
        if h_X != self.hash_X:
            self.X = X
            self.hash_X = h_X
            self.train(X)

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
        dim_input = batch.shape[1]
        # if network's architecture not specified, use (n, n/2, 5)
        if self.model is None:
            self.model = Autoencoder((dim_input, dim_input // 2, 2))

        self.model.cuda(self.gpu)
        # dirty hack
        model_dim_input = next(self.model.parameters()).size()[1]
        assert model_dim_input == dim_input
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        for _ in tqdm(range(self.nb_batches)):
            idx = np.random.randint(len(X), size=self.batch_size)
            batch = X[idx, :]
            P = (
                torch.tensor(make_P(batch, metric=self.distance_trajectory))
                .float()
                .cuda(self.gpu)
            )

            v = Variable(from_numpy(batch.astype(np.float32))).cuda(self.gpu)
            for epoch in range(self.n_iter):

                lat = self.model.encoder(v)
                output = self.model.decoder(lat)

                dist = criterion(output, v)
                kl = kl_divergence(lat, P, self.gpu)
                loss = dist + self.lambda_kl * kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.reco_err.append(dist.item())
                self.kls.append(kl.item())

        # disable gpu after training
        self.model.cpu()
        self.is_trained = True
