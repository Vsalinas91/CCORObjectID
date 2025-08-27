import os
import logging
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt

from skyfield.api import load
from .visualization.make_figure import plot_figure, set_image_yaw_state, scale_coordinates, reduce_vignette
from .utils.retrieve_data import (
    load_planetary_data,
    load_star_data,
    load_comet_data,
    subset_star_data,
    get_star_magnitude_mask,
    load_constellation_data,
)
from .utils.io import read_input, write_output, get_vignetting_func
from .utils.coordinate_transformations import (
    get_ccor_locations,
    get_ccor_locations_sunpy,
    get_comet_locations,
    get_star_names,
    get_ccor_observer,
    get_observer_subpoint,
)
from .utils.exceptions import CCORExitError


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def run_alg(inputs: list[Any], generate_figures: bool = False, write_output_files: bool = True) -> None:
    """
    Process CCOR L3 data to identify celestial objects within it's field of view (FOV).

    Objects are identified and catalogued within output files for reference or for validating
    star/planet fitting. The following objects are identified by the algorithm and may be used
    to construct a celestial map as seen from CCOR's perspective:

       -) Stars - uses the Hipparcos catalogue, ~100k stars and their brightness magnitudes and names
       -) Planets - uses AstroPy, and thus SPICE kernals for planetary ephemeris data retrieval
       -) Comets - uses the Minor Planet Center for comet ephemeris data retrieval
       -) Constellation Charts - edges for building constellations
    """
    # Load the timescale
    ts = load.timescale()

    # Define the ephemeris data:
    reference_bodies = load_planetary_data()
    earth = reference_bodies.earth
    sun = reference_bodies.sun
    stars = load_star_data()
    comets = load_comet_data()
    s_id: npt.NDArray[Any] = np.array(stars.index)

    # Format stars dataframe and define filter for getting only brightest stars:
    # Todo: handle this in retrieve_data.py
    get_star_magnitudes = get_star_magnitude_mask(stars)
    star_data = get_star_magnitudes.star_data
    bright_stars = get_star_magnitudes.bright_stars
    marker_size = get_star_magnitudes.marker_size

    # Get constellation data (contains names, and connecting star edges, or star catalogue IDs):
    constellations = load_constellation_data()

    # Get the vignetting function for masking out the pylon/occulter disc
    # in the object map.
    if generate_figures:
        vig_data = get_vignetting_func()

    for f in inputs[:]:
        # Initialize dicts to store the data
        # recombine when writing to output file.
        star_dict = {}
        planet_dict: dict[str, tuple[Any, Any]] = {}
        data_dict = {}
        comet_dict = {}

        # Get relevant data from the input L3 data file.
        logger.info(f"Running object identification for file: {os.path.basename(f)}")
        get_input_data = read_input(f, ts)
        data = get_input_data.image_data  # noqa: F841
        header = get_input_data.header  # noqa: F841
        wcs = get_input_data.WCS
        ccor_map = get_input_data.ccor_map
        t = get_input_data.time
        observation_time = get_input_data.obs_time
        end_time = get_input_data.end_time
        image_dims = wcs.array_shape
        logger.info(f"Identifying objects for observing time: {observation_time}")

        # Define the observer (approximate from GEO location for G19 if observer_geo is not set.):
        try:
            observer_geo = get_observer_subpoint(
                observation_time, header["EPHVEC_X"], header["EPHVEC_Y"], header["EPHVEC_Z"]
            )
        except KeyError:
            logger.error(
                "Invalid key in header for ephemeris positions. Make sure to map correct keys to function call."
            )
            continue
        logger.info(f"Observer subpoint (lon, lat): {observer_geo.lon}, {observer_geo.lat}")
        # make locs negative since skyfield uses positive values for western hemisphere.
        observer = get_ccor_observer(earth, -observer_geo.lat, -observer_geo.lon)

        # FOR STARS:
        # -----------
        logger.info("Getting star data from Hipparcos catalogue.")
        star_locations = get_ccor_locations(observer, t, wcs, star_data)
        # All stars
        s_x = star_locations.s_x
        s_y = star_locations.s_y
        # Now subset to the field of view only:
        logger.info("Acquiring stars within CCOR FOV at observation time.")
        star_object = subset_star_data(
            s_x=s_x,
            s_y=s_y,
            bright_stars=bright_stars,
            marker_size=marker_size,
            s_id=s_id,
            nx=image_dims[1],
            ny=image_dims[0],
        )
        good_sx_sub = star_object.stars_x
        good_sy_sub = star_object.stars_y
        good_markers_sub = star_object.markers
        good_star_ids = star_object.stars_ids
        # Get the star names from their ids:
        good_star_names = get_star_names(good_star_ids)
        logger.info(f"{len(good_sx_sub)} stars within FOV with magnitude > 7.")

        # FOR PLANETS/MOON:
        # ----------------
        logger.info("Getting planet(s)/moon within FOV.")
        planet_locations = get_ccor_locations_sunpy(ccor_map=ccor_map, observation_time=observation_time, wcs=wcs)

        # FOR COMETS:
        # -----------
        logger.info("Getting comet(s) within FOV.")
        get_comet, _, valid_pixels = get_comet_locations(
            comets=comets, sun=sun, ts=ts, observer=observer, observation_time=t, wcs=wcs
        )

        # FILE OUTGEST:
        # -------------
        logger.info("Building data dict for file outgest.")
        # Append the data
        comet_dict["comets"] = get_comet
        comet_dict["comet_locs"] = valid_pixels

        # Stars:
        star_dict["stars"] = (good_sx_sub, good_sy_sub)
        star_dict["star_markers"] = good_markers_sub.tolist()
        star_dict["star_ids"] = good_star_ids.tolist()
        star_dict["star_names"] = np.array(good_star_names, dtype=object).tolist()

        # Data:
        data_dict["date_obs"] = observation_time
        data_dict["date_end"] = end_time
        data_dict["parent_filename"] = Path(f).name

        combined_dict = {**data_dict, **comet_dict, **planet_dict, **star_dict}

        if write_output_files:
            try:
                logger.info("Writing to file.")
                write_output(observation_time, end_time, combined_dict)
            except CCORExitError:
                logger.exception("Cannot produce output file.")

        # PLOT IF TOGGLED:
        # ------------------
        if generate_figures:
            logger.info("Generting object map figure(s).")
            current_image_yaw_state = set_image_yaw_state(header["YAWFLIP"])
            image_frame_coords = scale_coordinates(header["CRPIX1"], header["CRPIX2"])
            crpix1 = image_frame_coords.crpix1
            crpix2 = image_frame_coords.crpix2

            # If the image is binned, bin the vignetting data too.
            if data.shape[0] != vig_data.shape[0]:
                vig_data = reduce_vignette(vig_data, current_image_yaw_state)

            # Generate figure for each time frame:
            plot_figure(
                data=data,
                date_obs=data_dict["date_obs"],
                date_end=data_dict["date_end"],
                vig_data=vig_data,
                comet_locs=comet_dict["comet_locs"],
                comet_name=comet_dict["comets"],
                star_locs=star_dict["stars"],
                star_ids=s_id,
                star_marker_size=good_markers_sub,
                all_star_x=s_x,
                all_star_y=s_y,
                constellations=constellations,
                planet_locs=planet_locations,
                crpix1=crpix1,
                crpix2=crpix2,
                naxis1=image_dims[1],
                naxis2=image_dims[0],
            )
