import os
import logging
from typing import Any
from datetime import timedelta, datetime
import numpy as np


from skyfield.api import load
from astropy.time import Time
import astropy.units as u

from .utils.io import read_input, write_sat_output
from .sat_utils.find_satellites import get_observer, propagate_satellite, propagate_sun
from .sat_utils.position_transformations import get_sat_angle, get_sat_az, do_sat_projection, angle_to_helioprojective

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

TFMT = "%Y-%m-%dT%H:%M:%S.%f"
strftime = datetime.strftime
fromisoformat = datetime.fromisoformat


def remove_duplicate_sat_entries(satellite_list: list[Any]) -> list[Any]:
    """
    Archived TLEs generally contain duplicate entries for the same satellite ID (not all).
    These duplicates usually define both a older and updated TLE for satellite with duplicate entries.
    This function removes the oldest TLE for a satellite with duplicate entries to ensure we
    only use the "latest" ephemeris data relative to date of the FITs file.
    """
    # Removes Duplicate Satellites:
    sat_names = [sat.name for sat in satellite_list[:]]
    duplicates = set()  # Create a set to store duplicate entries
    uniq_inds = []  # get the indices to access non-duplicated valid entries

    # Reverse the loop order since we want to get the last duplicated entry and not the first
    for i, sat_id in enumerate(sat_names[::-1]):
        if sat_id not in duplicates:
            # Ensures we get the last entry
            uniq_inds.append(i)
            duplicates.add(sat_id)

    # Get all valid satellites (no duplicates)
    logger.info(f"Satellite List records retained: {len(satellite_list)}.")

    # Reverse back to original order
    valid_satellites = [satellite_list[::-1][uniq_ind] for uniq_ind in uniq_inds][::-1]

    return valid_satellites


def run_satellite_id(
    inputs: list[Any],
    tle: str,
    search_radius: int | float = 30e3,
    fov_angle: int | float = 11,
    time_offset: int | float = 3,
    write_output_files: bool = True,
    observer_satellite: str = "GOES 19",
) -> None:
    """
    Retrieve candidate satellites, and their approximate pixel locations, within
    the FOV of the image being processed using concurrent two-line element (TLE)
    data for the date being processed.

    For this analysis, we identify all possible candidate satellites relative to:
       - DATE-BEG: beginning of image capture
       - DATE-AVG: average time of capture
       - DATE-END: end of image capture
    to try and identify satellites through the entire imaging sequence.

    The search radius is used to only search for valid satellites out to <search_radius> km
    from the position of the instrument.

    The <fov_angle> is used to only capture satellites within the visible FOV of the CCOR imagery.
    """
    # Get the satellite ephemeris data from TLE
    try:
        satellite_list = list(load.tle_file(tle))
    except Exception:
        logger.error("Invalid TLE data provided. Exiting program.")
    logger.info(f"Original Satellite List: {len(satellite_list)} entries")

    # Remove (possible) duplicate entries in archived TLE files.
    valid_satellites = remove_duplicate_sat_entries(satellite_list=satellite_list)

    # Start satellite search
    for f in inputs[:]:
        # Load the timescale
        ts = load.timescale()
        # DATA INGEST
        # Get relevant data from the input L3 data file.
        logger.info(f"Running object identification for file: {os.path.basename(f)}")
        get_input_data = read_input(f, ts)
        data = get_input_data.image_data  # noqa: F841
        header = get_input_data.header  # noqa: F841
        observer_map = get_input_data.ccor_map
        coordinate_frame = observer_map.coordinate_frame
        wcs = observer_map.wcs

        # TIMING
        # Define the image's observing time (start-time of observing sequence)
        # this is the reference frame we are interested in
        obstime = Time(header["DATE-OBS"], scale="utc")

        # Now define a leading and trailing time that preceeds/falls after the total
        # exposure time DATE-END - DATE-BEG
        obstime_leading = fromisoformat(header["DATE-OBS"]) - timedelta(minutes=time_offset)
        obstime_trailing = fromisoformat(header["DATE-END"]) + timedelta(minutes=time_offset)
        obstime_leading_fmt = strftime(obstime_leading, TFMT)
        obstime_trailing_fmt = strftime(obstime_trailing, TFMT)

        # Now define a list of times to iterate over to propgate all satellites.
        # this will allow for us to define a satellite streak across the image FOV
        # at the observing time: obstime that span a time range covered mainly by 2*time_offset +s
        # total exposure time (for CCOR this is 27 seconds)
        check_times = [
            obstime_leading_fmt,
            header["DATE-BEG"],
            header["DATE-AVG"],
            header["DATE-END"],
            obstime_trailing_fmt,
        ]
        astro_times = [Time(time, scale="utc") for time in check_times]
        julian_times = [ts.from_astropy(astro_time) for astro_time in astro_times]
        time_labels = [f"time{str(i).zfill(2)}" for i in range(len(check_times))]

        # Iterate over all times and project the satellite positions to those times relative
        # to our observer's observation time:
        for tidx, ccor_time in enumerate(julian_times):
            logger.info(f"Time: {check_times[tidx]}")

            # Get the Sun's position
            sun_position = propagate_sun(obstime)
            sun_vector_gcrs = sun_position.sun_vector_gcrs
            sun_vector_hpc = sun_position.sun_vector_hpc

            # Get the observer's position
            observer_sat = [sat for sat in valid_satellites if sat.name == observer_satellite]  # for CCOR
            observer_position = get_observer(
                header=header,
                obstime=obstime,
                satellite=observer_sat,
                coordinate_frame=coordinate_frame,
                observer_coordinate=observer_map,
                ts=ts,
                use_tle=True,
            )
            obs_vector_gcrs = observer_position.obs_vector_gcrs
            obs_vector_hpc = observer_position.obs_vector_hpc

            # Calculate Observer to sun distance:
            observer_sun_distance_meta = header["DSUN_OBS"] * 1e-3
            observer_sun_distance_calculated = np.abs(np.linalg.norm(obs_vector_gcrs) - np.linalg.norm(sun_vector_gcrs))
            distance_error = np.abs(observer_sun_distance_meta - observer_sun_distance_calculated)
            logger.info(f"Observer-Sun distance error: {distance_error} km or {(distance_error * u.km).to(u.au)}")

            for satellite in valid_satellites:
                if satellite.name == observer_satellite:
                    continue
                # Get the satellite position for each ccor_time
                sat_position = propagate_satellite(
                    satellite=satellite, time=ccor_time, obstime=obstime, coordinate_frame=coordinate_frame
                )
                sat_vector_gcrs = sat_position.sat_vector_gcrs
                sat_vector_hpc = sat_position.sat_vector_hpc

                # Get the angle between satellite-observer and observer-sun vectors [degrees]:
                sat_angle = get_sat_angle(sat_vector_gcrs, obs_vector_gcrs, sun_vector_gcrs)

                # Get the distances from satellite to observer [km]
                sat_dist = np.abs(np.linalg.norm(sat_vector_gcrs) - np.linalg.norm(obs_vector_gcrs))

                if sat_angle < fov_angle / 2:
                    # Do the projection of the satellite's 3D helioprojective coordinate onto the 2D observer plane
                    sat_project = do_sat_projection(
                        obs_coords=obs_vector_hpc, sat_coords=sat_vector_hpc, sun_coords=sun_vector_hpc
                    )
                    xproj = sat_project.xproj
                    yproj = sat_project.yproj
                    # Calculate the azimuth angle from the 2D observer plane's x-axis (solar west)
                    sat_az = get_sat_az(xproj, yproj) - 90

                    # Flip the azimuth for yaw flipped images
                    if "YAWFLIP" in header:
                        if header["YAWFLIP"] == 2:  # flipped image
                            logger.info("Adjusting HPC coordinate sign for yaw flipped image.")
                            factor = -1

                    # Calculate the heliprojective coordinates from the sat_angle:
                    #  note: units are in degrees, but can be converted to arcsec if
                    #        the CUNIT for the WCS is not degrees.
                    sat_position_hpc = angle_to_helioprojective(sat_angle, sat_az)
                    if header["CUNIT1"] == "deg":
                        unit = "deg"
                        astro_unit = u.deg
                    else:
                        unit = "arcsec"
                        astro_unit = u.arcsec

                    Tx = sat_position_hpc.Tx
                    Ty = sat_position_hpc.Ty

                    if unit == "arcsec":
                        Tx = (Tx * u.deg).to(u.arcsec).value
                        Ty = (Ty * u.deg).to(u.arcsec).value

                    # Now, calculate the pixel location using the input file's WCS:
                    hpc_coord = np.array([factor * Tx, factor * Ty]) * astro_unit
                    sat_pix = wcs.all_world2pix(hpc_coord, 0)  # origin is = 0 if image origin is bottom left corner
                    xpix = sat_pix[0][0]
                    ypix = sat_pix[0][1]

                    logger.info(
                        f"Satellite {satellite.name} is {sat_angle} from boresight "
                        + f"with an azimuthal position of {sat_az} relative to the observing plane. "
                        + f"Calculated pixel position is {xpix}, {ypix}."
                    )

        if write_output_files:
            write_sat_output(header["DATE-OBS"], header["DATE-END"], sat_dict)
