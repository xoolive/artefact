import torch.nn as nn
import numpy as np
from torch import nn, optim


def build_net(dims):
    ret = []
    for n1, n2 in zip(dims[:-1], dims[1:]):
        ret.append(nn.Linear(n1, n2))
        ret.append(nn.ReLU())
    return ret


def build_encoder(dims):
    return build_net(dims)


def build_decoder(dims):
    return build_net(dims)[:-1] + [nn.Sigmoid()]


class Autoencoder(nn.Module):
    def __init__(self, layers):
        """Build a new encoder using the architecture specified with
    [arch_encoder] and [arch_decoder].
        """

        super().__init__()
        if layers[0] != layers[-1]:
            arch_encoder = layers
            arch_decoder = tuple(reversed(layers))
        else:
            latent_index = np.argmin(layers)
            arch_encoder = layers[: i + 1]
            arch_decoder = layers[i:]

        self.encoder = nn.Sequential(*build_encoder(arch_encoder))

        arch_decoder = (
            list(reversed(arch_encoder))
            if arch_decoder is None
            else arch_decoder
        )

        self.decoder = nn.Sequential(*build_decoder(arch_decoder))

    def forward(self, x, **kwargs):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
