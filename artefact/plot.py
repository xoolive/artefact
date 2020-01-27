"""
utility module to plot clusters from:
https://github.com/lbasora/sectflow
"""
from itertools import cycle, islice
from random import sample

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import Axes3D, proj3d
from tqdm.autonotebook import tqdm
from traffic.data import airports
from traffic.drawing import EuroPP, Lambert93, PlateCarree, countries, rivers
from traffic.drawing.markers import atc_tower

from .clustering import get_latent


def anim_to_html(anim):
    plt.close(anim._fig)
    return anim.to_html5_video()


animation.Animation._repr_html_ = anim_to_html


C = [
    "r",
    "g",
    "blue",
    "y",
    "black",
    "cyan",
    "violet",
    "pink",
    "brown",
    "magenta",
    "turquoise",
    "orange",
    "peru",
    "palevioletred",
] * 5


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Annotation3D(Annotation):
    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self, s, xy=(0, 0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy = (xs, ys)
        Annotation.draw(self, renderer)


def annotate3D(ax, s, *args, **kwargs):
    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)


def plot_trajs(t, sector, proj=Lambert93()):
    n_clusters_ = int(1 + t.data.cluster.max())

    #  -- dealing with colours --
    color_cycle = cycle(
        ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf"]
        + ["#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00"]
    )

    colors = list(islice(color_cycle, n_clusters_))
    colors.append("#aaaaaa")  # color for outliers, if any

    # -- dealing with the grid --

    nb_cols = 5
    nb_lines = (1 + n_clusters_) // nb_cols + (((1 + n_clusters_) % nb_cols) > 0)

    def ax_iter(axes):
        if len(axes.shape) == 1:
            yield from axes
        if len(axes.shape) == 2:
            for ax_ in axes:
                yield from ax_

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(
            nb_lines, nb_cols, subplot_kw=dict(projection=proj), figsize=(15, 25),
        )

        for cluster, ax_ in tqdm(
            zip(range(t.data.cluster.min(), n_clusters_), ax_iter(ax))
        ):
            ax_.add_feature(countries())
            ax_.add_feature(rivers())

            tc = t.query(f"cluster == {cluster}")
            len_tc = len(tc)
            tc = tc[sample(tc.flight_ids, min(50, len(tc)))]
            tc.plot(ax_, color=colors[cluster])
            vr = tc.data.vertical_rate.mean()
            alt = tc.data.altitude.mean() // 100
            evolution = "=" if abs(vr) < 200 else "↗" if vr > 0 else "↘"
            ax_.set_title(f"{alt:.0f}FL{evolution}\nlen cluster:{len_tc}")

            if sector is not None:
                ax_.set_extent(sector)
                sector.plot(ax_, lw=2)


def clusters_plot2d(
    sector, t, nb_samples, projection, scaler=None, plot_trajs=False, plot_clust=None,
):
    with plt.style.context("traffic"):
        fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
        ax.add_feature(countries())
        ax.add_feature(rivers())
        ax.set_extent(
            tuple(
                x - 0.5 + (0 if i % 2 == 0 else 1) for i, x in enumerate(sector.extent)
            )
        )
        sector.plot(ax, lw=5)
        clust_ids = sorted(t.data.cluster.unique())
        if plot_clust is None:
            plot_clust = clust_ids
        for cid in plot_clust:
            tc = t.query(f"cluster=={cid}")
            if plot_trajs:
                for flight in tc:
                    lat = list(flight.data["latitude"])
                    lon = list(flight.data["longitude"])
                    if cid != -1:
                        ax.plot(
                            lon,
                            lat,
                            color=C[cid],
                            transform=PlateCarree(),
                            lw=1,
                            alpha=0.5,
                        )
                    else:
                        ax.plot(
                            lon,
                            lat,
                            color="grey",
                            transform=PlateCarree(),
                            lw=0.5,
                            alpha=0.5,
                        )
            if cid != -1:
                cent = tc.centroid(
                    nb_samples, projection=projection, transformer=scaler
                ).data
                lat = list(cent["latitude"])
                lon = list(cent["longitude"])
                ax.plot(lon, lat, color=C[cid], transform=PlateCarree(), lw=5)
                ax.arrow(
                    lon[-5],
                    lat[-5],
                    lon[-1] - lon[-5],
                    lat[-1] - lat[-5],
                    color=C[cid],
                    transform=PlateCarree(),
                    lw=3,
                    head_width=0.1,
                    head_length=0.1,
                )


def clusters_plot3d(
    sector,
    t,
    nb_samples,
    projection,
    scaler=None,
    plot_trajs=False,
    plot_clust=None,
    video=True,
):
    coords = np.stack(sector.flatten().exterior.coords)
    sector_Lon = [coords[k][0] for k in range(len(coords))]
    sector_Lat = [coords[k][1] for k in range(len(coords))]
    Upper = [layer.upper * 100 for layer in sector]
    Lower = [layer.lower * 100 for layer in sector]
    lonMin = min(sector_Lon)
    lonMax = max(sector_Lon)
    latMin = min(sector_Lat)
    latMax = max(sector_Lat)
    upper = max(Upper)
    lower = min(Lower)
    if upper == np.inf:
        upper = 45000

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # fig = plt.figure()
        # ax = plt.axes(projection="3d")
        ax.set_zlim(bottom=0, top=upper)
        ax.set_xlim(lonMin, lonMax)
        ax.set_ylim(latMin, latMax)

        ax.plot(
            sector_Lon,
            sector_Lat,
            [lower for k in sector_Lon],
            color="#3a3aaa",
            lw=2,
            alpha=0.5,
        )
        ax.plot(
            sector_Lon,
            sector_Lat,
            [upper for k in sector_Lon],
            color="#3a3aaa",
            lw=2,
            alpha=0.5,
        )
        for dot in zip(sector_Lon, sector_Lat):
            ax.plot(
                [dot[0], dot[0]],
                [dot[1], dot[1]],
                [lower, upper],
                color="#3a3aaa",
                lw=2,
                alpha=0.5,
            )

        clust_ids = sorted(t.data.cluster.unique())
        if plot_clust is None:
            plot_clust = clust_ids

        L = [
            sum([l == cid for l in t.data["cluster"]]) / len(t.data["cluster"])
            for cid in clust_ids
        ]

        for cid in plot_clust:
            tc = t.query(f"cluster=={cid}")
            if plot_trajs:
                for flight in tc:
                    lon = list(flight.data["longitude"])
                    lat = list(flight.data["latitude"])
                    alt = list(flight.data["altitude"])
                    cid = flight.data.cluster_id.iloc[0]
                    if cid != -1 and L[cid] >= 0.01:
                        ax.plot(lon, lat, alt, color=C[cid], lw=1, alpha=1)
                    else:
                        ax.plot(lon, lat, alt, color="grey", lw=1, alpha=0.5)

            l = L[cid]
            if l < 0.05:
                lw = 3
            elif l < 0.1:
                lw = 5
            else:
                lw = 7
            if cid != -1 and L[cid] >= 0.01:
                cent = tc.centroid(
                    nb_samples, projection=projection, transformer=scaler
                ).data
                lon = list(cent["longitude"])
                lat = list(cent["latitude"])
                alt = list(cent["altitude"])
                ax.plot(lon, lat, alt, color=C[cid], lw=lw)
                a = Arrow3D(
                    [lon[-5], lon[-1]],
                    [lat[-5], lat[-1]],
                    [alt[-5], alt[-1]],
                    mutation_scale=20,
                    lw=lw,
                    arrowstyle="-|>",
                    color=C[cid],
                )
                annotate3D(
                    ax,
                    s=str(cid),
                    xyz=(lon[-1], lat[-1], alt[-1]),
                    fontsize=25,
                    xytext=(-3, 3),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    color=C[cid],
                )
                ax.add_artist(a)
        if video:

            def animate(i):
                ax.view_init(20 + 5 * np.sin(i / 20), -60)
                return []

            return animation.FuncAnimation(
                fig, animate, frames=90, interval=200, blit=True
            )
        else:
            plt.show()


def plot_latent_and_trajs(
    t, lat, savefig, plot_clusters=False, airport="LSZH", runway=None
):
    if runway is not None:
        subset = t.query(f"runway == '{runway}' and initial_flow != 'N/A'")
    else:
        subset = t.query("initial_flow != 'N/A'")

    df = pd.DataFrame.from_records(
        [
            {"flight_id": id_, "x": x, "y": y}
            for (id_, x, y) in zip(list(f.flight_id for f in t), lat[:, 0], lat[:, 1],)
        ]
    )
    cols = ["flight_id", "simple", "initial_flow"]
    if plot_clusters:
        cols += ["cluster"]
    stasts = df.merge(subset.data[cols].drop_duplicates())

    with plt.style.context("traffic"):
        text_style = dict(
            verticalalignment="top",
            horizontalalignment="right",
            fontname="Ubuntu",
            fontsize=18,
            bbox=dict(facecolor="white", alpha=0.6, boxstyle="round"),
        )

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        fig = plt.figure(figsize=(15, 7.5))
        ax = fig.add_subplot(121)
        m = fig.add_subplot(122, projection=EuroPP())

        m.add_feature(
            countries(
                edgecolor="white", facecolor="#d9dadb", alpha=1, linewidth=2, zorder=-2
            )
        )
        m.outline_patch.set_visible(False)
        m.background_patch.set_visible(False)

        airports[airport].point.plot(
            m,
            shift=dict(units="dots", x=-15, y=-15),
            marker=atc_tower,
            s=300,
            zorder=5,
            text_kw={**text_style},
        )

        if plot_clusters:
            gb = "cluster"
        else:
            gb = "initial_flow"

        for (flow, d), color in zip(stasts.groupby(gb), colors):
            d.plot.scatter(x="x", y="y", ax=ax, color=color, label=flow, alpha=0.4)

            subset.query(f'{gb} == "{flow}"')[:50].plot(
                m, color=color, linewidth=1.5, alpha=0.5
            )

        ax.legend(prop=dict(family="Ubuntu", size=18))
        ax.grid(linestyle="solid", alpha=0.5, zorder=-2)

        # ax3.xaxis.set_tick_params(pad=20)
        ax.set_xlabel("1st component on latent space", labelpad=10)
        ax.set_ylabel("2nd component on latent space", labelpad=10)

        fig.savefig(savefig)


def plot_loss(loss, re_loss=None, kl_loss=None):
    plt.figure(1)
    plt.subplot(131)
    plt.plot(loss)
    plt.title("loss_evolution")
    if re_loss is not None:
        plt.subplot(132)
        plt.plot(re_loss)
        plt.title("re_evolution")
    if kl_loss is not None:
        plt.subplot(133)
        plt.plot(kl_loss)
        plt.title("kl_evolution")


def plot_latent(X, model, device):
    lat = get_latent(X, model, device)
    plt.scatter(lat[:, 0], lat[:, 1], s=10)


def dur_dist_plot(dur_dist):
    return (
        alt.Chart(dur_dist)
        .transform_density(
            "duration", as_=["duration", "density"], extent=[0, 70], groupby=["cluster"]
        )
        .mark_area(orient="horizontal")
        .encode(
            y="duration:Q",
            color="cluster:N",
            x=alt.X(
                "density:Q",
                stack="center",
                impute=None,
                title=None,
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
            ),
            column=alt.Column(
                "cluster:N",
                header=alt.Header(
                    titleOrient="bottom", labelOrient="bottom", labelPadding=0,
                ),
            ),
        )
        .properties(width=100)
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )
