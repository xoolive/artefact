"""
utility module to plot clusters from:
https://github.com/lbasora/sectflow
"""
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import Axes3D, proj3d
from tqdm.autonotebook import tqdm
from traffic.drawing import Lambert93, PlateCarree, countries, rivers


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
    nb_lines = (1 + n_clusters_) // nb_cols + (
        ((1 + n_clusters_) % nb_cols) > 0
    )

    def ax_iter(axes):
        if len(axes.shape) == 1:
            yield from axes
        if len(axes.shape) == 2:
            for ax_ in axes:
                yield from ax_

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(
            nb_lines,
            nb_cols,
            subplot_kw=dict(projection=proj),
            figsize=(15, 25)
        )

        for cluster, ax_ in tqdm(zip(range(-1, n_clusters_), ax_iter(ax))):
            ax_.add_feature(countries())
            ax_.add_feature(rivers())

            tc = t.query(f"cluster == {cluster}")
            tc.plot(ax_, color=colors[cluster])
            vr = tc.data.vertical_rate.mean()
            evolution = '=' if abs(vr) < 200 else '↗' if vr > 0 else '↘'
            ax_.set_title(f"v_rate:{vr:.0f}FL\nlen cluster:{len(tc)}")

            if sector is not None:
                ax_.set_extent(sector)
                sector.plot(ax_, lw=2)


def clusters_plot2d(
    sector,
    t,
    nb_samples,
    projection,
    scaler=None,
    plot_trajs=False,
    plot_clust=None,
):
    with plt.style.context("traffic"):
        fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
        ax.add_feature(countries())
        ax.add_feature(rivers())
        ax.set_extent(
            tuple(
                x - 0.5 + (0 if i % 2 == 0 else 1)
                for i, x in enumerate(sector.extent)
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
