import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from traffic.core import Traffic

from .ae import Autoencoder
from .clustering import AutoencoderTSNE


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

    ae_tsne = AutoencoderTSNE(
        gpu=gpu,
        model=model,
        pretrained_path=pretrained_path,
        algo_clustering=algo_clustering,
    )

    t_c = t.clustering(
        nb_samples=None,
        features=list_features,
        clustering=ae_tsne,
        transform=MinMaxScaler(feature_range=(-1, 1)),
    ).fit_predict()

    re, scores = ae_tsne.score_samples()
    t_c_re = pd.DataFrame.from_records(
        [dict(flight_id=f.flight_id, re=re) for f, re in zip(t_c, re)]
    )
    t_c = t_c.merge(t_c_re, on="flight_id")
    d = {"flight_id": "nunique", "re": ["mean", "min", "max"]}

    if scores is not None:
        t_c_scores = pd.DataFrame.from_records(
            [dict(flight_id=f.flight_id, score=score) for f, score in zip(t_c, scores)]
        )
        t_c = t_c.merge(t_c_scores, on="flight_id")
        d["score"] = ["mean", "min", "max"]

    print(t_c.groupby(["cluster"]).agg(d))
    t_c.to_pickle(to_pickle)

    return ae_tsne, t_c


def duration_cumdist_cluster(t):
    durations = pd.DataFrame.from_records(
        [
            dict(flight_id=f.flight_id, duration=d)
            for f, d in zip(t, [f.duration.seconds // 60 for f in t])
        ]
    )
    return (
        t.merge(durations, on="flight_id")
        .cumulative_distance(False)
        .eval(max_workers=10)
        .groupby("cluster")
        .agg({"flight_id": "nunique", "duration": "mean", "cumdist": "mean"})
    )


def duration_cumdist_flight(t):
    durations = pd.DataFrame.from_records(
        [
            dict(flight_id=f.flight_id, duration=d)
            for f, d in zip(t, [f.duration.seconds // 60 for f in t])
        ]
    )
    return (
        t.merge(durations, on="flight_id")
        .cumulative_distance(False)
        .eval(max_workers=10)
        .groupby("flight_id")
        .mean()
        .reset_index()
    )[["flight_id", "cluster", "duration", "cumdist"]]
