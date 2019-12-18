import pickle

import logging
import numpy as np
import pytorch_lightning as pl
import torch
from types import SimpleNamespace
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.cluster import DBSCAN
from torch import from_numpy, nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .autoencoder import Autoencoder
from .utils import kl_divergence, make_P


class AutoencoderTSNE_PL:
    def __init__(
        self,
        *,
        gpus=[0],
        model=None,
        learning_rate=0.001,
        weight_decay=0.01,
        lambda_kl=0.05,
        epochs=10,
        algo_clustering=DBSCAN(eps=0.3, min_samples=7),
        batch_size=100,
        distance_trajectory="euclidean",
        train_percent=1.0,
        accumulate_grad=1,
    ):
        self.gpus = gpus
        self.model = model
        self.algo_clustering = algo_clustering
        self.epochs = epochs
        self.is_trained = False
        self.X = None
        self.hash_X = None
        self.criterion = nn.MSELoss()
        self.train_percent = train_percent
        self.accumulate_grad = accumulate_grad

        self.hparams = SimpleNamespace(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lambda_kl=lambda_kl,
            distance_trajectory=distance_trajectory,
            batch_size=batch_size,
        )

        if len(self.gpus) > 1:
            self.distributed_backend = "ddp"
        else:
            self.distributed_backend = None

        # Reproducibility
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        np.random.seed(0)

    def fit(self, X):
        h_X = hash(X.tobytes())
        if h_X != self.hash_X:
            self.X = X
            self.hash_X = h_X
            self.train(X)

        v = from_numpy(X.astype(np.float32))
        lat = self.model.encoder(v).detach().numpy()
        self.labels_ = self.algo_clustering.fit_predict(lat)

    def to_pickle(self, filename):
        """Save the current AutoencoderTSNE in a pickle file named 'filename'
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def score_samples(self, X):
        """ Returns a numpy array containing the reconstruction error associated with each flight
        """
        v = from_numpy(X.astype(np.float32))
        output = self.model(v)
        return nn.MSELoss(reduction="none")(output, v).sum(1).detach().numpy()

    def train(self, X):
        logging.info("start training")
        dim_input = X.shape[1]
        if self.model is None:
            self.model = Autoencoder((dim_input, dim_input // 2, 2))

        model_dim_input = next(self.model.parameters()).size()[1]
        assert model_dim_input == dim_input
        trainer = Trainer(
            gpus=self.gpus,
            distributed_backend=self.distributed_backend,
            early_stop_callback=None,
            checkpoint_callback=ModelCheckpoint(
                filepath="./checkpoints", save_best_only=False
            ),
            train_percent_check=self.train_percent,
            max_nb_epochs=self.epochs,
            accumulate_grad_batches=self.accumulate_grad,
            default_save_path="./",
        )
        model_l = AutoencoderTSNEL(self.X, self.model, self.hparams)
        trainer.fit(model_l)
        self.model = model_l.model
        # disable gpu after training
        self.model.cpu()
        self.is_trained = True


class AutoencoderTSNEL(pl.LightningModule):
    def __init__(self, X, model, hparams):
        super(AutoencoderTSNEL, self).__init__()
        self.X = X
        self.model = model
        self.hparams = hparams
        self.criterion = nn.MSELoss()

    @pl.data_loader
    def train_dataloader(self):
        dataset = TensorDataset(torch.Tensor(self.X))
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=not self.use_ddp,
            sampler=DistributedSampler(dataset) if self.use_ddp else None,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def loss(self, x, lat, y):
        device = x.get_device()
        P = torch.tensor(make_P(x.cpu(), metric=self.hparams.distance_trajectory)).to(
            device
        )
        re = self.criterion(x, y)
        kl = kl_divergence(lat, P, device)
        return re + self.hparams.lambda_kl * kl, re, kl

    def training_step(self, batch, batch_nb):
        x = batch[0]
        lat, output = self.model(x)
        loss, re, kl = self.loss(x, lat, output)
           
        output = {
            "loss": loss,
            "progress_bar": {},
            "log": {"loss": loss, "re": re, "kl": kl},
        }
        return output
