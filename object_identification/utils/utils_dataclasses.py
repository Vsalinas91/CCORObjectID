from dataclasses import dataclass
from typing import Any
import numpy.typing as npt
from skyfield.vectorlib import VectorFunction

from astropy.io import fits
from astropy.wcs import WCS
from sunpy import map as smap
from skyfield.timelib import Timescale


# DATA INGEST:
# ------------
@dataclass(frozen=True, kw_only=True)
class GetData:
    image_data: npt.NDArray[Any]
    header: fits.Header
    WCS: WCS
    ccor_map: smap.GenericMap
    time: Timescale
    obs_time: str
    end_time: str


# DATA RETRIEVAL:
# ---------------
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


@dataclass(frozen=True, kw_only=True)
class GetReferenceBodies:
    earth: VectorFunction
    sun: VectorFunction
