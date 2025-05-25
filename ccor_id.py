from .utils.retrieve_data import load_planetary_data, load_star_data, load_comet_data
from .utils.io import read_input, write_output
from .utils.coordinate_transformations import (
    get_ccor_locations,
    get_ccor_locations_sunpy,
    get_comet_locations,
    get_star_names,
)

import os

from astropy.wcs import WCS
from astropy.time import Time
from sunpy import map as smap

from skyfield.api import Star, load
import skyfield.api as sf


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
    star_data = Star.from_dataframe(stars)
    limiting_magnitude = 7.0
    bright_stars = stars.magnitude <= limiting_magnitude * 2.5  # 1.5 for best effect
    magnitude = stars["magnitude"][bright_stars]
    marker_size = (0.5 + limiting_magnitude - magnitude) ** 2.0

    # Define the observer (approximate from GEO location for G19):
    observer_latitude = 0
    observer_longitude = 75.2
    observer = earth + sf.wgs84.latlon(observer_latitude, observer_longitude)  # Define observer location

    # For now, just initialize empty lists - will look into a
    # more effective solution later (maybe dict-like object)
    data = []
    all_comets = []
    all_valid_pixels = []
    all_stars = []
    all_marker_sizes = []
    all_moon_pixels = []
    all_merc_pixels = []
    all_venus_pixels = []
    all_jupiter_pixels = []
    all_saturn_pixels = []
    all_mars_pixels = []
    all_neptune_pixels = []
    all_uranus_pixels = []
    all_star_ids = []
    all_star_names = []
    date_obs = []
    for f in inputs[:1]:  # 10]:
        print(f"Identifying objects for file: {os.path.basename(f)}")
        data, header = read_input(f)

        # Define the WCS for the Celestial Coordinate system
        wcs = WCS(header, key="A")

        # For transformations relative to CCOR's coordinate system
        ccor_map = smap.Map(data, header, key="A")

        # Set the observation time (assuming you can get this from the FITS header)
        observation_time = header["DATE-OBS"]
        t = ts.from_astropy(Time(observation_time))

        # FOR STARS:
        # -----------
        s_x, s_y, s_distance = get_ccor_locations(observer, t, wcs, star_data)
        # Divide by two for plotting of L3 data
        good_sx = s_x[bright_stars]
        good_sy = s_y[bright_stars]
        # Now subset to the field of view only:
        star_mask = (
            (good_sx * 2 <= 2048) & (good_sx > 0) & (good_sy * 2 <= 1920) & (good_sy > 0)
        )  # multiply by two so they are found within original image bounds
        good_sx_sub = good_sx[star_mask]
        good_sy_sub = good_sy[star_mask]
        good_markers_sub = marker_size[star_mask]
        good_star_ids = s_id[bright_stars][star_mask]

        # STAR NAMES:
        # ---------
        good_star_names = get_star_names(good_star_ids)

        # FOR PLANETS/Moon:
        # ----------------
        planet_locations = get_ccor_locations_sunpy(ccor_map, observation_time, wcs)

        # FOR COMETS:
        # -----------
        valid_pixels, get_comet, get_distance = get_comet_locations(comets, sun, ts, observer, t, wcs)

        # Append the data
        all_comets.append(get_comet)
        all_valid_pixels.append(valid_pixels)
        all_stars.append((good_sx_sub, good_sy_sub))
        all_marker_sizes.append(good_markers_sub)
        all_star_ids.append(good_star_ids)
        all_star_names.append(good_star_names)

        # Planetary
        all_moon_pixels.append(planet_locations["moon"])
        all_merc_pixels.append(planet_locations["mercury"])
        all_venus_pixels.append(planet_locations["venus"])
        all_jupiter_pixels.append(planet_locations["jupiter"])
        all_saturn_pixels.append(planet_locations["saturn"])
        all_mars_pixels.append(planet_locations["mars"])
        all_neptune_pixels.append(planet_locations["neptune"])
        all_uranus_pixels.append(planet_locations["uranus"])
        data.append(data)
        date_obs.append(observation_time)

    # Now output the data:
    write_output(
        inputs,
        date_obs,
        all_comets,
        all_valid_pixels,
        all_moon_pixels,
        all_merc_pixels,
        all_venus_pixels,
        all_mars_pixels,
        all_jupiter_pixels,
        all_saturn_pixels,
        all_neptune_pixels,
        all_uranus_pixels,
        all_stars,
        get_star_names,
        all_star_ids,
    )

    # Plot if desired:
    # make_plots()
