from skyfield.api import load, Star, load_file
from skyfield.data import hipparcos, mpc, stellarium
import os

from dataclasses import dataclass
from typing import Any
import numpy.typing as npt
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.parent


@dataclass(frozen=True, kw_only=True)
class GetStarsSubset:
    stars_x: npt.NDArray[Any]
    stars_y: npt.NDArray[Any]
    markers: npt.NDArray[Any]
    stars_ids: npt.NDArray[Any]


@dataclass(frozen=True, kw_only=True)
class GetStarMags:
    star_data: npt.NDArray[Any]
    limiting_magnitude: float
    bright_stars: npt.NDArray[Any]
    magnitude: npt.NDArray[Any]
    marker_size: npt.NDArray[Any]


def load_planetary_data():
    """
    Load in planetary ephemeris data.
    """
    ephemeris = load_file(os.path.join(CURRENT_DIR, "static_required/de421.bsp"))
    return (ephemeris["earth"], ephemeris["sun"])


def load_star_data():
    """
    Load the Hipparcos start catalogue data.
    """
    try:
        with load.open(os.path.join(CURRENT_DIR, "static_required/hip_main.dat")) as f:
            return hipparcos.load_dataframe(f)
    except Exception:
        print("Could not load star catalogue - try different source.")
        return None


def get_star_magnitude_mask(stars, limiting_magnitude=7.0) -> GetStarMags:
    """
    Get a subset of the star catalogue based on the magnitude (limiting_magnitude)
    and give resulting marker sizes based on brightness for plotting.
    """
    star_data = Star.from_dataframe(stars)
    bright_stars = stars.magnitude <= limiting_magnitude * 2.5  # 1.5 for best affect
    magnitude = stars["magnitude"][bright_stars]
    marker_size = (0.5 + limiting_magnitude - magnitude) ** 2.0
    return GetStarMags(
        star_data=star_data,
        bright_stars=bright_stars,
        limiting_magnitude=limiting_magnitude,
        magnitude=magnitude,
        marker_size=marker_size,
    )


def subset_star_data(
    s_x: npt.NDArray[Any],
    s_y: npt.NDArray[Any],
    bright_stars: npt.NDArray[Any],
    marker_size: npt.NDArray[Any],
    s_id: npt.NDArray[Any],
    nx: int = 2048,
    ny: int = 1920,
) -> GetStarsSubset:
    """
    Subset the star catalogue to the CCOR FOV.
    """
    # Get brightest stars first using bright_stars masked defined in ccor_id.py
    good_sx = s_x[bright_stars]
    good_sy = s_y[bright_stars]

    # Subset to the FOV
    fov_mask = (good_sx * 2 <= nx) & (good_sx > 0) & (good_sy * 2 <= ny) & (good_sy > 0)
    good_sx_sub = good_sx[fov_mask].tolist()
    good_sy_sub = good_sy[fov_mask].tolist()
    good_markers_sub = marker_size[fov_mask]
    good_star_ids = s_id[bright_stars][fov_mask]

    return GetStarsSubset(stars_x=good_sx_sub, stars_y=good_sy_sub, markers=good_markers_sub, stars_ids=good_star_ids)


def load_comet_data():
    """
    Load the comet data.
    """
    try:
        with load.open(os.path.join(CURRENT_DIR, "static_required/CometEls.txt")) as f:
            comets = mpc.load_comets_dataframe(f)
            # Resort the dataframe:
            return (
                comets.sort_values("reference")
                .groupby("designation", as_index=False)
                .last()
                .set_index("designation", drop=False)
            )
    except Exception:
        print("Could not load comet data - try different source.")
        return None


def load_constellation_data():
    """
    Load the constellation data - major constellations.
    """
    const_url = os.path.join(CURRENT_DIR, "static_required/constellationship.fab")
    with load.open(const_url) as f:
        return stellarium.parse_constellations(f)
