from collections import defaultdict

from .utils.retrieve_data import (
    load_planetary_data,
    load_star_data,
    load_comet_data,
    subset_star_data,
    get_star_magnitude_mask,
)
from .utils.io import read_input, write_output
from .utils.coordinate_transformations import (
    get_ccor_locations,
    get_ccor_locations_sunpy,
    get_comet_locations,
    get_star_names,
    get_ccor_observer,
)

import os
from skyfield.api import load


def run_alg(inputs):
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
    earth, sun = load_planetary_data()
    stars = load_star_data()
    comets = load_comet_data()
    s_id = stars.index

    # Format stars dataframe and define filter for getting only brightest stars:
    get_star_magnitudes = get_star_magnitude_mask(stars)
    star_data = get_star_magnitudes.star_data
    bright_stars = get_star_magnitudes.bright_stars
    marker_size = get_star_magnitudes.marker_size

    # Define the observer (approximate from GEO location for G19):
    observer = get_ccor_observer(earth)

    # For now, just initialize empty lists - will look into a
    # more effective solution later (maybe dict-like object)
    star_dict = defaultdict(list)
    planet_dict = defaultdict(list)
    data_dict = defaultdict(list)
    comet_dict = defaultdict(list)

    for f in inputs[:1]:  # 10]:
        print(f"Identifying objects for file: {os.path.basename(f)}")
        get_input_data = read_input(f, ts)
        data = get_input_data.image_data
        wcs = get_input_data.WCS
        ccor_map = get_input_data.ccor_map
        t = get_input_data.time
        observation_time = get_input_data.obs_time

        # FOR STARS:
        # -----------
        s_x, s_y, s_distance = get_ccor_locations(observer, t, wcs, star_data)
        # Now subset to the field of view only:
        star_data = subset_star_data(s_x, s_y, bright_stars, marker_size, s_id)
        good_sx_sub = star_data.stars_x
        good_sy_sub = star_data.stars_y
        good_markers_sub = star_data.markers
        good_star_ids = star_data.stars_ids
        # Get the star names from their ids:
        good_star_names = get_star_names(good_star_ids)

        # FOR PLANETS/Moon:
        # ----------------
        planet_locations = get_ccor_locations_sunpy(ccor_map, observation_time, wcs)

        # FOR COMETS:
        # -----------
        valid_pixels, get_comet, get_distance = get_comet_locations(comets, sun, ts, observer, t, wcs)

        # Append the data
        comet_dict["comets"].append(get_comet)
        comet_dict["comet_locs"].append(valid_pixels)

        # Stars:
        star_dict["stars"].append((good_sx_sub, good_sy_sub))
        star_dict["star_markers"].append(good_markers_sub)
        star_dict["star_ids"].append(good_star_ids)
        star_dict["star_names"].append(good_star_names)

        # Planetary
        keys = ["mercury", "venus", "moon", "mars", "jupiter", "saturn", "uranus", "neptune"]
        for k in keys:
            planet_dict[k].append(planet_locations[k])

        # Data:
        data_dict["data"].append(data)
        data_dict["date_obs"].append(observation_time)

    # Now output the data:
    write_output(
        inputs,
        planet_dict,
        star_dict,
        data_dict,
        comet_dict,
        get_star_names,
    )

    # Plot if desired:
    # make_plots()
