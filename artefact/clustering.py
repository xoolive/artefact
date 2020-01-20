import logging
import torch
from sklearn.cluster import DBSCAN
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
        learning_rate=0.001,
        weight_decay=0.01,
        lambda_kl=0.05,
        nb_iterations=500,
        algo_clustering=DBSCAN(eps=0.3, min_samples=7),
        batch_size=1000,
        distance_trajectory="euclidean",
        savedir=None,
    ):
        self.device = get_device(gpu)
        self.model = model
        self.algo_clustering = algo_clustering
        self.nb_iterations = nb_iterations
        self.savedir = savedir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_kl = lambda_kl
        self.distance_trajectory = distance_trajectory
        self.batch_size = batch_size

    def fit(self, X):
        self.X = X
        self.train()
        lat = self.get_latent()
        self.labels_ = self.algo_clustering.fit_predict(lat)

    def get_latent(self):
        return get_latent(self.X, self.model, self.device)

    def score_samples(self):
        """ Returns a numpy array containing the reconstruction error associated with each flight
        """
        self.model.eval()
        self.model.to(self.device)
        v = torch.as_tensor(self.X, dtype=torch.float, device=device)
        with torch.no_grad():
            output = self.model(v)
            return nn.MSELoss(reduction="none")(output, v).sum(1).cpu().numpy()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self):
        dim_input = self.X.shape[1]
        if self.model is None:
            self.model = Autoencoder((dim_input, dim_input // 2, 2))

        model_dim_input = next(self.model.parameters()).size()[1]
        assert model_dim_input == dim_input

        self.model, self.loss = train(
            self.model,
            self.X,
            device=self.device,
            nb_iterations=self.nb_iterations,
            batch_size=self.batch_size,
            lambda_kl=self.lambda_kl,
            distance_trajectory=self.distance_trajectory,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.savedir is not None:
            torch.save(self.model.state_dict(), f"{self.savedir}/model.pth")

