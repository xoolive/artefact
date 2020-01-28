import argparse
import os
from pathlib import Path

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from traffic.core import Traffic

from artefact import Autoencoder, AutoencoderTSNE


def main(args):
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    traffic = Traffic.from_file(args.data)
    list_features = ["track_unwrapped", "longitude", "latitude", "altitude"]

    nb_samples = len(traffic[0])
    nb_features = len(list_features)

    algo_clustering = AutoencoderTSNE(
        gpu=args.gpu,
        model=Autoencoder((nb_samples * nb_features, 32, 8, 2)),
        learning_rate=args.learning_rate,
        weight_decay=args.learning_rate,
        lambda_kl=args.lambda_kl,
        nb_iterations=args.nb_iterations,
        batch_size=args.batch_size,
        algo_clustering=DBSCAN(eps=0.06, min_samples=20),
        distance_trajectory="euclidean",  # delta_max
        savepath=f"{args.savepath}/model.pth",
    )

    t_tsne = traffic.clustering(
        nb_samples=None,  # nb_samples,
        features=list_features,
        clustering=algo_clustering,
        transform=MinMaxScaler(feature_range=(-1, 1)),
    ).fit_predict()

    t_tsne.to_parquet(f"{args.savepath}/t_tsne.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", dest="data", type=Path, default="data/lszh.parquet")
    parser.add_argument("-o", dest="savepath", type=Path, default="results")
    parser.add_argument("-g", dest="gpu", type=int, default=0)
    parser.add_argument("-it", dest="nb_iterations", type=int, default=100)
    parser.add_argument("-bs", dest="batch_size", type=int, default=1000)
    parser.add_argument("-la", dest="lambda_kl", type=float, default=0.05)
    parser.add_argument("-lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd", dest="weight_decay", type=float, default=1e-5)
    args = parser.parse_args()
    main(args)
