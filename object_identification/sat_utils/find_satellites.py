import os
import logging
from dataclasses import dataclass
from typing import Any
import numpy.typing as npt
from skyfield.vectorlib import VectorFunction
from sunpy.map.mapbase import GenericMap

import numpy as np
from astropy.coordinates import SkyCoord, CartesianRepresentation
import astropy.units as u

from . import position_transformations

get_angle = position_transformations.get_angle
get_angular_positions = position_transformations.get_angular_positions

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class GetAllSatellites:
    all_sat_names: npt.NDArray[Any]  # catalogue sattelite names in TLE
    all_sat_coords: npt.NDArray[Any]  # coordinates/positions of satellites
    all_sun_coords: npt.NDArray[Any]  # sun coordinates at obstime
    all_ccor_coords: npt.NDArray[Any]  # ccor coordinates at obstime
    all_earth_coords: npt.NDArray[Any]  # earth coordinates at obstime
    all_sat_pos: npt.NDArray[Any]  # satellite positions (VectorFunction) objects
    goes_sat_coords: npt.NDArray[Any]  # GOES-19 satellite coords


@dataclass(frozen=True, kw_only=True)
class GetCandidateSatellites:
    get_angle_in_fov: npt.NDArray[Any]  # satellite-sun angle (similar to sun-earth, and/or sun-moon)
    get_dist: npt.NDArray[Any]  # distance from satellite to observer
    get_sat_id: npt.NDArray[Any]  # satellite ids within FOV
    get_tlabel: npt.NDArray[Any]  # time labels for mapping values/locations to corresponding time stamp
    get_angle_locs: npt.NDArray[Any]  # approximation to 2D locations of satellite in FOV/image
    get_sat_collection: list[Any]  # satellite vector function objects for valid satellites
    get_sat_pos: npt.NDArray[Any]  # satellite vector positions for reference


##########################################################################
# We Want the Positions for All possible Times During the Image Capture: #
##########################################################################
def get_all_positions_for_times(
    astro_times: list[Any],
    j_times: list[Any],
    earth: VectorFunction,
    sun: VectorFunction,
    ccor_map: GenericMap,
    valid_sat: list[Any],
    use_gcrs: bool = False,
) -> GetAllSatellites:
    """
    Get the Sun, CCOR, Earth, and satellite positions for
    all times reported in a CCOR data product file's header.

    This ensures that we capture all possible satellites positions projected
    to DATE-BEG=DATE-OBS, DATE-AVG, DATE-END.

    The inputs are:

        -) astro_times = astropy Time objects for date-obs, date-avg, date-end
        -) j_times = jdate times for date-obs, date-avg, date-end for predicting positions
        -) earth = earth ephemeris object
        -) sun = sun ephemeris object
        -) ccor_map = ccor_map object
        -) valid_sat = all valid satellites (no duplicates)

    There are several options for the coordinate transformations using toggle use_gcrs:

        -) Use GCRS will cast all coordinates into the Geocentric Celestial Reference System
        -) Not using GCRS will default the coordinates to being cast into a Geocentric Earth Equatorial System
           which is more ideal when dealing with satellite positions relative to other celestial bodies.

    Further, all coordinates are relative to the Earth's position, but this can be cast into a barycentric frame
    by simply subtracting the Earth positions--at each time--from the Sun position, and removed from how th
    satellite position vector is defined at the observation time.

    Returns:
        -) all_sat_names = all satellite names being checked
        -) all_sat_coords = all satellite coordinates for all times
        -) all_sun_coords = all sun coordinates for all times
        -) all_ccor_coords = all ccor coordinates for all times
        -) all_earth_coords = all earth coordinates for all times
        -) all_sat_pos = all satellite positions at observation time, can be converted

    """
    # Init all lists
    all_sat_names = []  # Satellite names
    all_sat_coords = []  # satellite coordinates in GCRS/GEE cartesian
    all_sun_coords = []  # Sun coordinates in GCRS/GEE cartesian
    all_ccor_coords = []  # CCOR coordinates in GCRS/GEE cartesian
    all_earth_coords = []  # Earth coordinates in GCRS/GEE cartesian
    all_sat_pos = []  # Satellite position vector relative to Earth

    # Iterate over the file times:
    for at, t in zip(astro_times, j_times):
        logger.info(f"Getting satellite data for time: {at.isot}")

        # Get Earth and Sun Locations at Observation Time
        # ---------------------------------------------
        earth_loc = earth.at(t)
        sun_loc = sun.at(t)  # noqa: F841

        # Observe the sun from the Earth to get it's position relative to Earth at the observatory time.
        # and get it's position vector
        s = earth_loc.observe(sun).apparent()
        if use_gcrs:
            # use GCRS
            sunx, suny, sunz = s.position.km
            earx, eary, earz = earth_loc.position.km
        else:
            # use GEE
            srep = CartesianRepresentation(s.position.km * u.km)
            scoord = SkyCoord(srep, frame="geocentricearthequatorial", representation_type="cartesian", obstime=at)

            erep = CartesianRepresentation(earth_loc.position.km * u.km)
            ecoord = SkyCoord(erep, frame="geocentricearthequatorial", representation_type="cartesian", obstime=at)

            sunx, suny, sunz = scoord.cartesian.xyz.to(u.km).to_value()
            earx, eary, earz = ecoord.cartesian.xyz.to(u.km).to_value()

        # Get the CCOR Observer Coordinates:
        # ------------------------------------------------
        # Convert the CCOR observation location
        if use_gcrs:
            ccor_loc = ccor_map.observer_coordinate.transform_to("gcrs")
        else:
            ccor_loc = ccor_map.observer_coordinate.transform_to("geocentricearthequatorial")
        # Make into a SkyCoord object
        ccor_loc = SkyCoord(
            CartesianRepresentation(
                ccor_loc.cartesian.x.to(u.km), ccor_loc.cartesian.y.to(u.km), ccor_loc.cartesian.z.to(u.km)
            ),
            frame="geocentricearthequatorial",
            representation_type="cartesian",
            obstime=at,
        )

        # As with the earth satellite objects, we must also add the earth position to the CCOR position
        # vector to account for a barycentric reference frame
        # [DISABLE ADDING EARTH if not earth relative] for ccor AND sun vectors
        ccor_x = ccor_loc.cartesian.x.to(u.km).to_value() + earx
        ccor_y = ccor_loc.cartesian.y.to(u.km).to_value() + eary
        ccor_z = ccor_loc.cartesian.z.to(u.km).to_value() + earz

        # Commented out, but remove earth position if casting into barycentric frame
        # sunx -= earx
        # suny -= eary
        # sunz -= earz

        # Get satellite positions and ids
        # --------------------------------------------------
        sat_xyzs = []  # Satellite xyzs, same as coords
        sat_name = []  # The satellite name, not NORAD Cat ID
        sat_coords = []  # Satellite coordinates (actually the same thing)
        sat_id = []  # satellite catalogue number
        sat_pos = []  # satellite position at observation time
        for i, sat in enumerate(valid_sat):
            # Define the position relative to Earth in a barycentric reference frame
            # ensures that all bodies are in the same reference frame
            position = (earth + sat).at(t).position.km  # skip earth reference for now
            # position = sat.at(t).position.km # remove earth if desired

            # Get the satellites cartesian coordinates for transforming into a SkyCoord object
            rep = CartesianRepresentation(position * u.km)
            if use_gcrs:
                # If using GCRS, insert frame here
                sat_coord = SkyCoord(rep, frame="gcrs", obstime=at)
            else:
                # Else we use GEE
                # p = (earth+sat).at(t).position.km
                # v = (earth+sat).at(t).velocity.km_per_s
                # teme_v = CartesianDifferential(v*u.km/u.s)
                # teme_p = CartesianRepresentation(p*u.km)
                # teme = TEME(teme_p.with_differentials(teme_v), obstime=at)

                # rep = teme.transform_to(frames.GeocentricEarthEquatorial(obstime=at)).cartesian
                sat_coord = SkyCoord(
                    rep, frame="geocentricearthequatorial", obstime=at, representation_type="cartesian"
                )

            # Append all relevant satellite data
            sat_coords.append(sat_coord)
            sat_xyzs.append(sat_coord)
            sat_name.append(sat.name)
            sat_id.append(sat.model.satnum)
            sat_pos.append((earth + sat).at(t))

        # Define the CCOR To Sun Line-of-Sight (LOS):
        # ---------------------------------------------------
        # Define CCOR-to-Sun Line of Sight (LOS)
        # Note: because it's CCOR relative, need to adjust angles by 180 if
        # plotting the 3D satellite locations relative to CCOR's FOV
        pos = np.sqrt((ccor_x - sunx) ** 2.0 + (ccor_y - suny) ** 2.0 + (ccor_z - sunz) ** 2.0)
        az = np.arctan2((ccor_y - suny), (ccor_x - sunx))  # 180 - az in degrees if plotting grid
        el = np.arccos((ccor_z - sunz) / pos)  # also needs to be adjusted if plotting  # noqa: F841
        sc_angle = np.rad2deg(np.arccos((sunz - ccor_z) / pos))  # angle to sun's position
        logger.info(
            f"Sun is at an inclination angle of {sc_angle} from "
            + f"x-axis at time {t.utc_datetime()} and {az} from horizontal."
        )

        all_sat_names.append(sat_name)
        all_sat_coords.append(sat_coords)
        all_sun_coords.append((sunx, suny, sunz))
        all_ccor_coords.append((ccor_x, ccor_y, ccor_z))
        all_earth_coords.append((earx, eary, earz))
        all_sat_pos.append(sat_pos)

    # Find goes position:
    goes_sat_coord = np.array(all_sat_coords)[np.where(np.array(all_sat_names) == "GOES 19")]

    return GetAllSatellites(
        all_sat_names=np.array(all_sat_names),
        all_sat_coords=np.array(all_sat_coords),
        all_sun_coords=np.array(all_sun_coords),
        all_ccor_coords=np.array(all_ccor_coords),
        all_earth_coords=np.array(all_earth_coords),
        all_sat_pos=np.array(all_sat_pos),
        goes_sat_coords=np.array(goes_sat_coord),
    )


def get_satellites_in_fov(
    tlabels: list[Any],
    all_ccor_coords: npt.NDArray[Any],
    all_sun_coords: npt.NDArray[Any],
    all_sat_coords: npt.NDArray[Any],
    all_sat_names: npt.NDArray[Any],
    all_sat_pos: npt.NDArray[Any],
    fov_angle: int | float = 11,
    radius_search: int | float = 30e3,
) -> GetCandidateSatellites:
    """
    Identify all satellites in the CCOR FOV within 11 degrees of the boresight and
    retrieve their cartesian locations, angular positions from the boresight, distances from
    CCOR at their respective times:

    Returns:
       - get_angle_in_fov = the angle relative to the SUN-CCOR vector, boresight within the FOV
       - get_dist = distnace from satellite to ccor within the FOV
       - get_sat_id = name of satellites in FOV
       - get_tlabel = get the time label for the position
       - get_angle_locs = get the azimuthal and inclination angles for plotting
       - get_sat_pos = get the satellite positions in the FOV
       - get_sat_collation = get satellites locations within the search radius
    """
    # Initialize arrays of type object to store lists of varying sizes
    get_angle_in_fov = np.zeros([len(tlabels)], dtype="object")
    get_dist = np.zeros([len(tlabels)], dtype="object")
    get_sat_id = np.zeros([len(tlabels)], dtype="object")
    get_tlabel = np.zeros([len(tlabels)], dtype="object")
    get_angle_locs = np.zeros([len(tlabels)], dtype="object")
    get_sat_pos = np.zeros([len(tlabels)], dtype="object")
    get_sat_collection = []

    # Iterate over all 3 timestamps in CCOR file:
    for tidx in np.arange(all_sat_names.shape[0]):
        # Initialize lists to write to array objects for each time
        get_close_points = []
        get_close_ids = []
        valid_angles = []
        valid_ids = []
        valid_tlabel = []
        valid_dists = []
        valid_angle_locs = []
        valid_sat_pos = []

        # Get the CCOR, SUN, and [optional] GOES19 positions
        ccor_coord = all_ccor_coords[tidx]
        sun_coord = all_sun_coords[tidx]

        ccor_x = ccor_coord[0]
        ccor_y = ccor_coord[1]
        ccor_z = ccor_coord[2]

        sunx = sun_coord[0]
        suny = sun_coord[1]
        sunz = sun_coord[2]

        # Iterate over the satellites for each time
        for id, ss, sat_pos in zip(all_sat_names[tidx], all_sat_coords[tidx], all_sat_pos[tidx]):
            # Don't look at GOES-19 since CCOR is onboard GOES 19
            if id == "GOES 19":
                continue

            # Get the satellite positions
            s = ss.cartesian.xyz.to(u.km).to_value()
            xrel = s[0]
            yrel = s[1]
            zrel = s[2]
            # Search over valid radius box
            if (
                (np.abs(xrel - ccor_x) < radius_search)
                & (np.abs(yrel - ccor_y) < radius_search)
                & (np.abs(zrel - ccor_z) < radius_search)
            ):
                # Shouldn't be an issue, but if NOT GOES-19, then proceed
                if "GOES 19" not in id:
                    # Get the distance from ccor to the satellite
                    spos = np.sqrt((ccor_x - xrel) ** 2.0 + (ccor_y - yrel) ** 2.0 + (ccor_z - zrel) ** 2.0)
                    # Get distance from satellite to sun
                    spos_srel = np.sqrt(  # noqa: F841
                        ((xrel - ccor_x) - (sunx - ccor_x)) ** 2.0
                        + ((yrel - ccor_y) - (suny - ccor_y)) ** 2.0
                        + ((zrel - ccor_z) - (sunz - ccor_z)) ** 2.0
                    )

                    # Get the satellite angle relative to the sun center
                    sat_angle = get_angle(
                        np.array([xrel, yrel, zrel]), np.array([ccor_x, ccor_y, ccor_z]), np.array([sunx, suny, sunz])
                    )

                    angular_locs = get_angular_positions(
                        np.array([xrel, yrel, zrel]),
                        np.array([ccor_x, ccor_y, ccor_z]),
                        np.array([sunx, suny, sunz]),
                    )

                    # Get all satellite locations in the radius box
                    get_close_points.append(((xrel - ccor_x), (yrel - ccor_y), (zrel - ccor_z)))
                    get_close_ids.append(id)

                    # Check if satellite is within boresight FOV:
                    if (sat_angle) <= fov_angle / 2:
                        # Correct for yaw flip:
                        valid_angles.append(sat_angle)
                        # valid_locs.append(locs)
                        valid_ids.append(id)
                        valid_dists.append(spos)
                        valid_tlabel.append(tlabels[tidx])
                        valid_angle_locs.append((angular_locs[0].tolist(), angular_locs[1].tolist()))
                        valid_sat_pos.append(sat_pos)
                        logger.info(
                            f"{id} has a SAT_ANGLE of {sat_angle} from the boresight | Angular Locations (x, y):"
                            + f" {angular_locs} | Distance from CCOR is {spos} km."
                        )

        # Store in lists
        get_angle_in_fov[tidx] = valid_angles
        get_dist[tidx] = valid_dists
        get_sat_id[tidx] = valid_ids
        get_tlabel[tidx] = valid_tlabel
        get_angle_locs[tidx] = valid_angle_locs
        get_sat_collection.append(get_close_points)
        get_sat_pos[tidx] = valid_sat_pos

    return GetCandidateSatellites(
        get_angle_in_fov=get_angle_in_fov,
        get_dist=get_dist,
        get_sat_id=get_sat_id,
        get_tlabel=get_tlabel,
        get_angle_locs=get_angle_locs,
        get_sat_collection=get_sat_collection,
        get_sat_pos=get_sat_pos,
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
