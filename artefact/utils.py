from traffic.core import Traffic
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .clustering import AutoencoderTSNE
from .ae import Autoencoder


def pretrained_clust(
    traffic_file,
    list_features,
    algo_clustering,
    model,
    pretrained_path,
    to_pickle,
    gpu=0,
):
    t = Traffic.from_file(traffic_file)
    nb_samples = len(t[0])
    nb_features = len(list_features)

    ae_tsne = AutoencoderTSNE(
        gpu=gpu,
        model=model,
        pretrained_path=pretrained_path,
        algo_clustering=algo_clustering,
    )

    t_tsne = t.clustering(
        nb_samples=None,
        features=list_features,
        clustering=ae_tsne,
        transform=MinMaxScaler(feature_range=(-1, 1)),
    ).fit_predict()

    re = ae_tsne.score_samples()
    re = MinMaxScaler(feature_range=(0, 1)).fit_transform(re.reshape(-1, 1)).flatten()
    t_tsne_re = pd.DataFrame.from_records(
        [dict(flight_id=f.flight_id, re=re) for f, re in zip(t_tsne, re)]
    )
    t_tsne_re = t_tsne.merge(t_tsne_re, on="flight_id")

    t_tsne_re.to_pickle(to_pickle)
    print(
        t_tsne_re.groupby(["cluster"]).agg(
            {"flight_id": "nunique", "re": ["mean", "min", "max"]}
        )
    )
    return ae_tsne, t_tsne_re
