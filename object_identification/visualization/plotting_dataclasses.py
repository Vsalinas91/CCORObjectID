from matplotlib.collections import PathCollection
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class ImageCenterData:
    scaling: int
    crpix1: int | float
    crpix2: int | float


@dataclass(frozen=True, kw_only=True)
class CelestialBodyPlot:
    body_plot: PathCollection | None
    body_name: str | None


@dataclass(frozen=True, kw_only=True)
class ConstellationData:
    get_const_name: list[Any]
    get_const_lines: list[Any]
