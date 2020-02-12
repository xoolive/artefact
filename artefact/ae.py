import torch.nn as nn
import numpy as np
from torch import nn, optim


def build_net(dims,reluout=True):
    ret = []
    n = len(dims[:-1])
    for i,(n1, n2) in enumerate(zip(dims[:-1], dims[1:])):
        ret.append(nn.Linear(n1, n2))
        if i<n-1 or reluout:
            ret.append(nn.ReLU())
    return ret


def build_encoder(dims,reluout=True):
    return build_net(dims,reluout)


def build_decoder(dims):
    return build_net(dims)[:-1] + [nn.Sigmoid()]


class Autoencoder(nn.Module):
    def __init__(self, layers,reluout=True):
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

        self.encoder = nn.Sequential(*build_encoder(arch_encoder,reluout))

        arch_decoder = (
            list(reversed(arch_encoder))
            if arch_decoder is None
            else arch_decoder
        )
        decode = build_decoder(arch_decoder)
        if not reluout:
            decode = [nn.ReLU()]+decode
        self.decoder = nn.Sequential(*decode)
        print("encoder",self.encoder)
        print("decoder",self.decoder)

    def forward(self, x, **kwargs):
        lat = self.encoder(x)
        output = self.decoder(lat)
        return lat, output
