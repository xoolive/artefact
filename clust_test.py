import argparse
from pathlib import Path

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from traffic.core import Traffic

from artefact import Autoencoder, AutoencoderTSNE


def main(args):
    traffic = Traffic.from_file(args.filename)
    list_features = ["track_unwrapped", "longitude", "latitude", "altitude"]

    nb_samples = len(traffic[0])
    nb_features = len(list_features)

    algo_clustering = AutoencoderTSNE(
        gpus=args.gpus,
        model=Autoencoder((nb_samples * nb_features, 32, 8, 2)),
        lambda_kl=args.lambda_kl,
        epochs=args.epochs,
        algo_clustering=DBSCAN(eps=0.06, min_samples=20),
        distance_trajectory="euclidean",  # delta_max
    )

    t_tsne = traffic.clustering(
        nb_samples=None,  # nb_samples,
        features=list_features,
        clustering=algo_clustering,
        transform=MinMaxScaler(feature_range=(-1, 1)),
    ).fit_predict()

    t_tsne.to_pickle("t_tsne.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", dest="filename", type=Path, default="lszh.pkl")
    parser.add_argument("-g", dest="gpus", type=str, default="0,1,2")
    parser.add_argument("-e", dest="epochs", type=int, default=100)
    parser.add_argument("-bs", dest="batch_size", type=int, default=100)
    parser.add_argument("-la", dest="lambda_kl", type=float, default=0.05)
    parser.add_argument("-lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd", dest="weight_decay", type=float, default=1e-5)
    parser.add_argument("-ag", dest="accumulated_grad", type=int, default=1)
    args = parser.parse_args()
    main(args)
