import os
import logging
from typing import Any

from skyfield.api import load
from astropy.time import Time


from .utils.retrieve_data import load_planetary_data
from .utils.io import read_input, write_sat_output
from .sat_utils.find_satellites import get_all_positions_for_times, get_satellites_in_fov

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


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
    write_output_files: bool = True,
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
    # Load the timescale
    ts = load.timescale()
    tlabels = ["date-beg", "date-avg", "date-end"]

    # Define the earth/sun ephemeris data:
    reference_bodies = load_planetary_data()
    earth = reference_bodies.earth
    sun = reference_bodies.sun

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
        # Get relevant data from the input L3 data file.
        logger.info(f"Running object identification for file: {os.path.basename(f)}")
        get_input_data = read_input(f, ts)
        data = get_input_data.image_data  # noqa: F841
        header = get_input_data.header  # noqa: F841
        ccor_map = get_input_data.ccor_map

        # Get the time(s) for which we'll search for satellites
        obs_time = Time(header["DATE-OBS"])
        avg_time = Time(header["DATE-AVG"])
        end_time = Time(header["DATE-END"])

        tobs = ts.from_astropy(obs_time)
        tavg = ts.from_astropy(avg_time)
        tend = ts.from_astropy(end_time)

        astro_times = [obs_time, avg_time, end_time]
        j_times = [tobs, tavg, tend]

        # Get Sun, Earth, CCOR, and Satellite Ephemeris Position Coordinates:
        obs_satellite_data = get_all_positions_for_times(astro_times, j_times, earth, sun, ccor_map, valid_satellites)

        # Identify Target Satellites within the FOV:
        candidates = get_satellites_in_fov(
            tlabels,
            obs_satellite_data.all_ccor_coords,
            obs_satellite_data.all_sun_coords,
            obs_satellite_data.all_sat_coords,
            obs_satellite_data.all_sat_names,
            obs_satellite_data.all_sat_pos,
            fov_angle=fov_angle,
            radius_search=search_radius,
        )  # noqa: F841

        sat_dict = {}.fromkeys(["satellite_name", "satellite_angle", "satellite_pos", "satellite_distance"])
        sat_dict["satellite_name"] = candidates.get_sat_id
        sat_dict["satellite_angle"] = candidates.get_angle_in_fov
        sat_dict["satellite_position"] = candidates.get_angle_locs
        sat_dict["satellite_distance"] = candidates.get_dist

        if write_output_files:
            write_sat_output(header["DATE-OBS"], header["DATE-END"], sat_dict)
