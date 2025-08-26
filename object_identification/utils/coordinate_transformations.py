from typing import Any
import numpy.typing as npt
from skyfield.vectorlib import VectorFunction
from astropy.wcs.wcs import WCS
from sunpy.map.mapbase import GenericMap
from pandas import DataFrame
from skyfield.timelib import Timescale

from astropy.coordinates import SkyCoord, get_body, EarthLocation
from astropy.time import Time

from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield.data import mpc
from skyfield.named_stars import named_star_dict
import skyfield.api as sf

from .utils_dataclasses import ObjectLocations


def get_ccor_locations(
    observer: VectorFunction, observation_time: str, wcs: WCS, objects: npt.NDArray[Any]
) -> ObjectLocations:
    """
    Get the pixel locations of the objects relative to CCOR's
    WCS.
    """
    # Get positions relative to observaation time and observer location
    object_positions = observer.at(observation_time).observe(objects)
    # Get the angular positions for converting to the WCS CCOR pixel world
    obj_ra, obj_dec, obj_distance = object_positions.radec()
    obj_x, obj_y = wcs.all_world2pix(obj_ra.degrees, obj_dec.degrees, 1)  # 1 for origin at 1
    return ObjectLocations(s_x=obj_x, s_y=obj_y, object_distance=obj_distance)


def get_ccor_locations_sunpy(ccor_map: GenericMap, observation_time: str, wcs: WCS) -> dict[str, tuple[Any, Any]]:
    """
    Get the pixel locations for planetary bodies using SunPy's Map object
    """
    ccor_itrs = ccor_map.observer_coordinate.transform_to("itrs")
    el = EarthLocation.from_geocentric(x=ccor_itrs.x, y=ccor_itrs.y, z=ccor_itrs.z)

    keys = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "moon"]
    planet_dict: dict[str, tuple[Any, Any]] = {}

    for key in keys:
        body = get_body(key, time=Time(observation_time), location=el)
        body_skycoord = SkyCoord(body.ra, body.dec, frame="icrs", unit="deg")
        body_pixel_x, body_pixel_y = wcs.world_to_pixel(body_skycoord)
        planet_dict[key] = (float(body_pixel_x) / 2, float(body_pixel_y) / 2)

    return planet_dict


def get_comet_locations(
    comets: DataFrame, sun: VectorFunction, ts: Timescale, observer: VectorFunction, observation_time: str, wcs: WCS
):
    """
    Get the comet pixel locations relative to CCOR's WCS and observation time.
    """
    # Want to build lists for each comet
    valid_pixels = []
    get_comet = []
    get_distance = []
    # Iterate over all comets
    for body in comets["designation"]:
        # Get the data for each comet and define it's orbit
        comet_row = comets.loc[body]
        orbit = sun + mpc.comet_orbit(comet_row, ts, GM_SUN)
        # Get the position relative to the observer
        comet_position = observer.at(observation_time).observe(orbit)
        comet_ra, comet_dec, distance = comet_position.radec()
        # Get the comet position
        comet_x, comet_y = wcs.all_world2pix(comet_ra.degrees, comet_dec.degrees, 1)  # 1 for origin at 1
        # Make sure it's withing the FOV bounds:
        if (comet_x <= 2048) & (comet_x > 0) & (comet_y <= 1920) & (comet_y > 0) & (distance.au < 2):
            get_comet.append(body)
            get_distance.append(distance)
            valid_pixels.append((comet_x / 2, comet_y / 2))

    return (get_comet, get_distance, valid_pixels)


def get_star_names(star_ids: list[int | float] | npt.NDArray[Any]) -> list[list[str]]:
    """
    From the star data, get the name corresponding to the catalogue ID (HIP ID)
    """
    get_names = []
    for hip_id in star_ids:
        star_names = hip_id_to_star_name(hip_id)
        get_names.append(star_names)
    return get_names


def hip_id_to_star_name(star_id: int | float) -> list[str]:
    """
    Converts a Hipparcos (HIP) ID to a star name.
    """
    return [name for name, hip_id in named_star_dict.items() if star_id == hip_id]


def get_ccor_observer(earth: VectorFunction) -> VectorFunction:
    """
    Define the observer location to do the coordinate transformations.
    """
    observer_latitude = 0
    observer_longitude = 75.2
    return earth + sf.wgs84.latlon(observer_latitude, observer_longitude)
