import logging
from time import time

import torch
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from .training import train


def get_device(gpu_number):
    return torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")


def get_latent(X, model, device):
    model.eval()
    model.to(device)
    v = torch.as_tensor(X, dtype=torch.float, device=device)
    with torch.no_grad():
        lat = model.encoder(v).cpu().numpy()
        return lat


class AutoencoderTSNE:
    def __init__(
        self,
        gpu=0,
        model=None,
        learning_rate=1e-3,
        weight_decay=1e-5,
        lambda_kl=0.05,
        nb_iterations=500,
        algo_clustering=DBSCAN(eps=0.3, min_samples=7),
        batch_size=1000,
        distance_trajectory="euclidean",
        pretrained_path=None,
        savepath=None,
    ):
        self.device = get_device(gpu)
        self.model = model
        self.algo_clustering = algo_clustering
        self.nb_iterations = nb_iterations
        self.savepath = savepath
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_kl = lambda_kl
        self.distance_trajectory = distance_trajectory
        self.batch_size = batch_size
        self.pretrained_path = pretrained_path
        if self.pretrained_path is not None:
            self.load_weights(self.pretrained_path)

    def fit(self, X):
        ti = time()
        self.X = X
        if self.pretrained_path is None:
            self.train()
        lat = self.get_latent()
        self.labels_ = self.algo_clustering.fit_predict(lat)
        tf = time()
        logging.info("Fit time:  ", tf - ti)

    def get_latent(self):
        return get_latent(self.X, self.model, self.device)

    def score_samples(self):
        """ Returns a numpy array containing the reconstruction error associated with each flight
        """
        self.model.eval()
        self.model.to(self.device)
        v = torch.as_tensor(self.X, dtype=torch.float, device=self.device)
        with torch.no_grad():
            _, output = self.model(v)
            re = nn.MSELoss(reduction="none")(output, v).sum(1).cpu().numpy()
            re = (
                MinMaxScaler(feature_range=(0, 1))
                .fit_transform(re.reshape(-1, 1))
                .flatten()
            )

        scores = None
        if isinstance(self.algo_clustering, GaussianMixture):
            scores = self.algo_clustering.score_samples(self.get_latent())
            scores = (
                MinMaxScaler(feature_range=(0, 1))
                .fit_transform(scores.reshape(-1, 1))
                .flatten()
            )

        return re, scores

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self):
        dim_input = self.X.shape[1]
        if self.model is None:
            self.model = Autoencoder((dim_input, dim_input // 2, 2))

        model_dim_input = next(self.model.parameters()).size()[1]
        assert model_dim_input == dim_input

        self.model, self.loss = train(
            model=self.model,
            X=self.X,
            device=self.device,
            nb_iterations=self.nb_iterations,
            batch_size=self.batch_size,
            lambda_kl=self.lambda_kl,
            distance_trajectory=self.distance_trajectory,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.savepath is not None:
            torch.save(self.model.state_dict(), f"{self.savepath}")
        return self.model, self.loss
