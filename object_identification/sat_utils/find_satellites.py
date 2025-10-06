import os
import logging
from dataclasses import dataclass
from typing import Any
import numpy as np
import numpy.typing as npt

from skyfield.framelib import itrs
from skyfield.timelib import Time, Timescale
from skyfield.api import EarthSatellite

from astropy.coordinates import SkyCoord, CartesianRepresentation, ICRS
import astropy.units as u
from astropy.time import Time as astroTime
from astropy.coordinates import get_body, get_body_barycentric
from astropy.io.fits import Header

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class GetSatellitePosition:
    sat_vector_gcrs: npt.NDArray[Any]
    sat_vector_hpc: npt.NDArray[Any]


@dataclass(frozen=True, kw_only=True)
class GetObserverPosition:
    obs_vector_gcrs: npt.NDArray[Any]
    obs_vector_hpc: npt.NDArray[Any]


@dataclass(frozen=True, kw_only=True)
class GetSunPosition:
    sun_vector_gcrs: npt.NDArray[Any]
    sun_vector_hpc: npt.NDArray[Any]


def propagate_satellite(
    satellite: EarthSatellite, time: float | Time, obstime: astroTime, coordinate_frame: SkyCoord
) -> GetSatellitePosition:
    """
    Using the observation time, propagate satellite positions to the time range
    specified for identifying candidate satellites in the FOV.

    For CCOR, suggest using 5 times for generating satellite "streaks" as
    often times identifying satellites for only the observation time will
    not work. This is due to how much satellites move within a 15 minute period
    depending on their proximity to the observer. Streaks help capture the general motion,
    and may ensure that candidate satellite positions in an image may be near or
    within the derived streak.
    """
    # Get geocentric International Celestial Reference System ICRS satellite position
    #   note: ICRS origin the solar system barycenter, but for Earth Satellites, this is
    #         relative to Earth.
    sat_geo = satellite.at(time)
    # For calculating the sat_angle
    sat_vector_gcrs = sat_geo.position.km

    # Now transform to a International Terrestrial Reference System ITRS frame for
    # eventually transforming our position to a heliocentric frame. ITRS is terrestrial,
    # and so it is geocentric
    #   note: This is used to do the 3D->2D projection onto the observer plane relative to Solar North.
    sat_vector_itrs = sat_geo.frame_xyz(itrs).m
    # Define a skycoord object for the ITRS coordinate. Here we use the obervation time
    # so that our heliocentric frame matches that provided in the metadata.
    sat_vector_itrs_skycoord = SkyCoord(CartesianRepresentation(sat_vector_itrs * u.m), frame="itrs", obstime=obstime)
    # Now cast the position into a heliocentric frame:
    sat_coord_helio = sat_vector_itrs_skycoord.transform_to(coordinate_frame).cartesian.xyz.to(u.km).value

    return GetSatellitePosition(
        sat_vector_gcrs=sat_vector_gcrs,
        sat_vector_hpc=sat_coord_helio,
    )


def propagate_sun(obstime: astroTime) -> GetSunPosition:
    """
    Using the observation time, calculate the Sun's barycentric ICRS coordinate position
    relative to Earth. We are effectively casting the Sun's position into the same reference frame
    as both our observer and the satelite(s).
    """
    sun_vector_gcrs = get_body("sun", time=obstime).cartesian.xyz.to(u.km).value
    sun_vector_hpc = np.array([0, 0, 0])  # heliocentric, the sun is at the origin
    return GetSunPosition(
        sun_vector_gcrs=sun_vector_gcrs,
        sun_vector_hpc=sun_vector_hpc,
    )


def get_observer(
    header: Header,
    obstime: astroTime,
    satellite: EarthSatellite,
    coordinate_frame: SkyCoord,
    observer_coordinate: SkyCoord,
    ts: Timescale,
    use_tle: bool = True,
) -> GetObserverPosition:
    """
    Get the observer position at the observation time in an ICRS coordinate system relative to Earth. This can be done
    two ways:
       1. Metadata: Use the header ephemeris vector position to define the observer ICRS coordinate.
       2. TLE: Using the same TLE, one can also retrieve the EarthSatellite object for the observer and use that
               to define it's position relative to other satllites. This is preferred as there is less of a chance
               for introducing timing offsets since this is coming from a consistent data source.
    """
    if not use_tle:
        # Use the metadata - this is prone to error given timing offsets bewteen the report position and
        # those in the TLE. This is used in place of using an OBSGEO coordinate in case such coordinates do not exist.
        try:
            # Earth's position is needed if we are using our EPHVEC EME2000 coordinate vector
            # to construct the observer position in GCRS.
            earth_barycentric = SkyCoord(
                CartesianRepresentation(get_body_barycentric("earth", obstime)), frame="icrs", obstime=obstime
            ).cartesian.xyz.to(u.km)
            # Observer ephemeris vector in EME2000 coords
            ephvec_x = header["EPHVEC_X"] * u.m  # Example EPHVEC X coordinate
            ephvec_y = header["EPHVEC_Y"] * u.m  # Example EPHVEC Y coordinate
            ephvec_z = header["EPHVEC_Z"] * u.m  # Example EPHVEC Z coordinate
            observer_gcrs = SkyCoord(
                x=ephvec_x, y=ephvec_y, z=ephvec_z, frame="itrs", obstime=obstime, representation_type="cartesian"
            ).transform_to(ICRS())
            # Get the GCRS/ICRS position; need logic to handle signage for when observer is on other side of earth.
            obs_vector_gcrs = earth_barycentric.value - observer_gcrs.cartesian.xyz.to(u.km).value
        except KeyError:
            try:
                # Use the predefined OBSGEO coordinates instead
                obsgeo_x = header["OBSGEO-X"]  # m
                obsgeo_y = header["OBSGEO-Y"]  # m
                obsgeo_z = header["OBSGEO-Z"]  # m
                obs_vector_gcrs = np.array([obsgeo_x, obsgeo_y, obsgeo_z])
            except KeyError:
                raise ValueError(
                    "No such metadata for OBSGEO coordinates - suggest using TLE data instead and trying again."
                )
        # Define the observer coordinate in units km from the obserer coordinate in the metaddata
        obs_coord_helio = observer_coordinate.cartesian.xyz.to(u.km).value
    else:
        # Use of the TLE is more reliable as we are certain to capture passing/neighboring satellites at the same times.
        # For the position from the TLE, want to use the lagged time--this will require more rigorous testing
        obs_geo = satellite.at(ts.from_astropy(Time(obstime)))
        obs_vector_gcrs = obs_geo.position.km
        # Now get the ITRS position
        obs_vector_itrs = obs_geo.frame_xyz(itrs).km
        # Define the heliocentric coordinate of the ITRS position vector.
        obs_coord_helio = (
            SkyCoord(CartesianRepresentation(obs_vector_itrs * u.km), frame="itrs", obstime=obstime)
            .transform_to(coordinate_frame)
            .cartesian.xyz.value
        )

    return GetObserverPosition(
        obs_vector_gcrs=obs_vector_gcrs,
        obs_vector_hpc=obs_coord_helio,
    )


def create_cone_mask(shape, center, radius, height, angle, direction, grid):
    """
    Creates a 3D cone mask.

    Args:
    shape: Tuple (x, y, z) representing the dimensions of the 3D array.
    center: Tuple (x, y, z) representing the center of the cone base.
    radius: Radius of the cone base.
    height: Height of the cone.
    angle: Angle of the cone (in radians), where 0 is a straight line and pi/2 is a flat circle.
    direction: Tuple (x, y, z) representing the direction of the cone's axis.

    Returns:
    A 3D NumPy array (boolean mask) representing the cone.
    """

    x, y, z = grid
    x0, y0, z0 = center
    dx, dy, dz = direction

    # Normalize direction
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm
    dy /= norm
    dz /= norm

    # Cone equation
    # Distance along the axis
    axis_dist = (x - x0) * dx + (y - y0) * dy + (z - z0) * dz

    # Distance from the axis
    perp_x = x - x0 - axis_dist * dx
    perp_y = y - y0 - axis_dist * dy
    perp_z = z - z0 - axis_dist * dz
    perp_dist = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)

    # Calculate cone radius at each point
    cone_radius = (radius / height) * axis_dist

    # Create mask
    mask = (axis_dist >= 0) & (axis_dist <= height) & (perp_dist <= cone_radius)
    return mask
