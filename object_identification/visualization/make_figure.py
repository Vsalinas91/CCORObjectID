import os
import logging
from pathlib import Path
from typing import Any
import numpy.typing as npt
from matplotlib.typing import ColorType


import numpy as np
from skimage.measure import block_reduce

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.legend import Legend

from .color_map import ccor_blue
from .plotting_dataclasses import ImageCenterData, CelestialBodyPlot, ConstellationData


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent
CMAP = ccor_blue()


def reduce_vignette(vig_data: npt.NDArray[Any], flip: int) -> npt.NDArray[Any]:
    return block_reduce(np.rot90(vig_data, flip), block_size=2, func=np.nanmean)


def set_image_yaw_state(yawflip: int | float) -> int:
    #    If yaw flipped, flip vignetting image
    if yawflip == 2:
        return 0
    else:
        return 2


def scale_coordinates(
    crpix1: int | float, crpix2: int | float, scale: bool = True, shift_x: int = -15, shift_y: int = 5
) -> ImageCenterData:
    # Determine if coords need to be scaled: This is for data who's CDELT's
    # have not been scaled so anything after May 9th, 2025
    scale = True
    scaling = 2 if scale else 1
    cx = crpix1 + shift_x
    cy = crpix2 + shift_y

    return ImageCenterData(scaling=scaling, crpix1=cx, crpix2=cy)


def plot_celstial_body(name: str, body_pixels: tuple[Any, Any], ax: Axes, color: ColorType) -> CelestialBodyPlot:
    """
    Plot any celestial body whose pixel locations are within the CCOR FOV.
    """
    body_plot = None
    body_name = None
    if len(body_pixels) >= 1:
        if body_pixels[0] is not None:
            if name in ["Moon", "moon"]:
                color = "w"
            body_name = name
            body_plot = ax.scatter(body_pixels[0], body_pixels[1], marker="o", color=color, s=200, edgecolor="w")

    return CelestialBodyPlot(body_plot=body_plot, body_name=body_name)


def setup_legends(
    data: list[Any], names: npt.NDArray[Any] | list[Any], title: str, ypos: int | float, axis: Axes
) -> Legend:
    """
    Routine to set up the object map legend.
    """
    legend = axis.legend(
        data,
        names,
        bbox_to_anchor=(1.05, ypos),
        loc="upper left",
        borderaxespad=0.0,
        facecolor="k",
        labelcolor="w",
        prop={"weight": "bold"},
        title=title,
    )
    plt.setp(legend.get_title(), color="w", weight="bold")

    return legend


def plot_constellations(
    constellations: list[Any],
    star_ids: npt.NDArray[Any],
    star_locs_x: npt.NDArray[Any],
    star_locs_y: npt.NDArray[Any],
    scaling: int,
    naxis1: int,
    naxis2: int,
    crpix1: int | float,
    crpix2: int | float,
    ax: Axes,
) -> ConstellationData:
    "Plot constellations from alls star locations."
    c_colors = plt.cm.gray(np.linspace(0.2, 1, len(constellations)))  # type: ignore[attr-defined]
    get_const_name = []
    get_const_lines = []
    for c_ind, (name, edges) in enumerate(constellations):
        cname = name
        for edge in edges:
            start = np.where(star_ids == edge[0])
            end = np.where(star_ids == edge[1])
            xs, ys = star_locs_x[start] * scaling, star_locs_y[start] * scaling
            xe, ye = star_locs_x[end] * scaling, star_locs_y[end] * scaling

            # For getting the right name
            xline = np.linspace(xs[0], xe[0], 50)
            yline = np.linspace(ys[0], ye[0], 50)
            rline = np.sqrt((xline - crpix1) ** 2.0 + (yline - crpix2) ** 2.0) < 560
            good_points = rline.tolist().count(True)

            if (
                any(xline * scaling > 0)
                & any(xline * scaling <= naxis1)
                & any(yline * scaling > 0)
                & any(yline * scaling <= naxis2 * 2)
                & (good_points > 0)
            ):
                valid_name = cname
                (const_lines,) = ax.plot(
                    [xs, xe], [ys, ye], linestyle="-", color=c_colors[c_ind], linewidth=1.5, zorder=-1, alpha=1
                )
            else:
                valid_name = None
                const_lines = None

            get_const_name.append(valid_name)
            get_const_lines.append(const_lines)

    return ConstellationData(get_const_lines=get_const_lines, get_const_name=get_const_name)


def plot_figure(
    data: npt.NDArray[Any],
    date_obs: str,
    date_end: str,
    vig_data: npt.NDArray[Any],
    comet_locs: list[tuple[Any, Any]],
    comet_name: list[str],
    star_locs: tuple[npt.NDArray[Any], npt.NDArray[Any]],
    star_ids: npt.NDArray[Any],
    star_marker_size: npt.NDArray[Any],
    all_star_x: npt.NDArray[Any],
    all_star_y: npt.NDArray[Any],
    constellations: list[Any],
    planet_locs: dict[str, tuple[Any, Any]],
    scaling: int,
    crpix1: int | float,
    crpix2: int | float,
    naxis1: int = 2048,
    naxis2: int = 1920,
    save_figures: bool = False,
) -> None:
    """
    Plot the L3 image along side the object map.
    """
    fig, ax = plt.subplots(1, 2, figsize=(17, 5))
    [axs.set_xticks([]) for axs in ax.flatten()]
    [axs.set_yticks([]) for axs in ax.flatten()]

    ax[1].set_facecolor("black")

    # --------------------------------
    # PLOT THE DATA and OVERLAY COMET
    # --------------------------------
    ax[0].pcolormesh(
        np.ma.MaskedArray(data, mask=~(vig_data > 0.01)),
        cmap=CMAP,
        norm=colors.PowerNorm(gamma=0.45, vmin=1e-12, vmax=5e-10),
    )
    ax[0].contourf(np.ma.MaskedArray(vig_data, mask=(vig_data > 0.01)), colors="black")

    # ---------------------------------
    # PLOT OBJECT MAP:
    # ---------------------------------
    # Plot comet locations
    n_comets = len(comet_locs)
    comet_cmap = plt.cm.Spectral(np.linspace(0, 1, n_comets))  # type: ignore[attr-defined]
    comet_plots = []
    comet_names = []
    for ncomet, p in enumerate(comet_locs):  # can be more than one, so iterate over them.
        comet_plot = ax[1].scatter(
            (p[0]) + 25,
            (p[1]) + 21,
            marker="H",
            color=comet_cmap[ncomet],
            edgecolor="w",
            s=90,
        )
        comet_plots.append(comet_plot)
        comet_names.append(comet_name[ncomet])

    # Plot the star map:
    ax[1].scatter(
        np.array(star_locs[0]),
        np.array(star_locs[1]),
        s=star_marker_size * 2,
        color="w",
        edgecolor="C0",
        marker="o",
        alpha=0.8,
        zorder=-12,
    )

    # Plot the planetary/lunar locations
    body_plots = []
    body_names = []
    planet_colors = plt.cm.jet(np.linspace(0, 1, len(planet_locs)))  # type: ignore[attr-defined]
    for index, (planet_name, locs) in enumerate(planet_locs.items()):
        plot_planet = plot_celstial_body(name=planet_name, body_pixels=locs, ax=ax[1], color=planet_colors[index])
        body_names.append(planet_name)
        body_plots.append(plot_planet.body_plot)

    # Plot Constellations:
    constellations_to_plot = plot_constellations(
        constellations, star_ids, all_star_x / 2, all_star_y / 2, scaling, naxis1, naxis2, crpix1, crpix2, ax[1]
    )

    # Only build legend using valid constellations if not None
    get_const_lines = constellations_to_plot.get_const_lines
    get_const_name = constellations_to_plot.get_const_name
    get_valid_seg = [get_const_lines[i] for i in range(len(get_const_lines)) if get_const_lines[i] is not None]
    get_valid_name = [get_const_name[i] for i in range(len(get_const_lines)) if get_const_lines[i] is not None]
    valid_indices = [np.where(name == np.array(get_valid_name))[0][0] for name in np.unique(get_valid_name)]

    # LEGENDS:
    # -------
    # Constellation(s)
    constellation_legend = setup_legends(
        data=np.array(get_valid_seg)[valid_indices].tolist(),
        names=np.array(get_valid_name)[valid_indices].tolist(),
        title="Constellation(s)",
        ypos=1,
        axis=ax[1],
    )
    # Planet(s)
    planet_legend = setup_legends(data=body_plots, names=body_names, title="Planet(s)", ypos=0.55, axis=ax[1])
    # Comet(s):
    comet_legend = setup_legends(data=comet_plots, names=comet_names, title="Comet(s)", ypos=0.125, axis=ax[1])

    ax[1].add_artist(constellation_legend)
    ax[1].add_artist(planet_legend)
    ax[1].add_artist(comet_legend)

    # ---------------------------------
    # FINAL PLOT SET UP AND FORMATTING
    # ---------------------------------
    # Overlay pylon/occulter disc on object map
    ax[1].contourf(np.ma.MaskedArray(vig_data, mask=vig_data > 0.01), colors="black", hatches=["//"])
    ax[1].contour(vig_data, levels=[0.009, 0.01], colors="white", linewidths=0.7, alpha=0.7)

    # plt.scatter(s_x//2, s_y//2)
    ax[1].set_xlim(0, 1024)
    ax[1].set_ylim(0, 960)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("Object Map", fontsize=15)
    ax[0].set_title(f"CCOR-1: {date_obs}", fontsize=14)

    if save_figures:
        obs_time_fmt = date_obs.replace("-", "").replace(":", "").split(".")[0]
        end_time_fmt = date_end.replace("-", "").replace(":", "").split(".")[0]
        file_tstamp = f"s{obs_time_fmt}Z_e{end_time_fmt}Z"
        out_dir = f"{obs_time_fmt.split('T')[0]}"

        try:
            os.makedirs(os.path.join(ROOT_DIR, f"figures/{out_dir}"), exist_ok=True)
        except OSError:
            logger.exception("Error creating data directory.")

        plt.savefig(
            os.path.join(ROOT_DIR, f"figures/{out_dir}/sci_ccor1-obj_g19_{file_tstamp}.png"),
            dpi=150,
            bbox_inches="tight",
        )

    plt.subplots_adjust(right=0.7)  # Adjust 'right' value as needed
    plt.show()
