from torch import nn
import itertools


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def build_conv1D(channels, kernel_sizes, strides, paddings):
    layers = []
    for (ci, co), ks, ksmp, s, p in zip(
        pairwise(channels), kernel_sizes[0], kernel_sizes[-1], strides, paddings
    ):
        layer = nn.Sequential(
            nn.Conv1d(ci, co, kernel_size=ks, stride=s, padding=p),
            nn.ReLU(),
            nn.MaxPool1d(ksmp),
        )
        layers.append(layer)
    # print(layers)
    return nn.Sequential(*layers)


def build_conv1DT(channels, kernel_sizes, strides, paddings, output_paddings):
    layers = []
    for (ci, co), ks, s, p, op in zip(
        pairwise(channels), kernel_sizes, strides, paddings, output_paddings
    ):
        layer = nn.Sequential(
            nn.ConvTranspose1d(
                ci, co, kernel_size=ks, stride=s, padding=p, output_padding=op
            ),
            nn.ReLU(),
        )
        layers.append(layer)
    # print(layers)
    return nn.Sequential(*layers)


class Conv1AE(nn.Module):
    def __init__(
        self,
        nb_features,
        nb_samples,
        channels,
        kernel_sizes,
        strides,
        paddings,
        output_paddings,
    ):
        super().__init__()
        self.nb_features, self.nb_samples = nb_features, nb_samples
        self.encoder = build_conv1D(
            channels, kernel_sizes[:-1], strides[0], paddings[0]
        )
        self.decoder = build_conv1DT(
            reversed(channels),
            kernel_sizes[-1],
            strides[-1],
            paddings[-1],
            output_paddings,
        )

    def forward(self, x):
        # input: (batch_size, nb_samples*nb_features) -> (batch_size, nb_features, nb_samples)
        print("input", x.shape)
        x = x.view(-1, self.nb_features).T.reshape(-1, self.nb_features, self.nb_samples)
        print("resized", x.shape)
        x = self.encoder(x)
        print("encoded", x.shape)
        x = self.decoder(x)
        print("decoded", x.shape)
        # print(x.view(-1, self.nb_features * self.nb_samples).shape)
        return x.view(-1, self.nb_features * self.nb_samples)
