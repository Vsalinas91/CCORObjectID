import json
import os
import datetime

from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from sunpy import map as smap


def read_input(input, ts):
    """Read fits file and return data and header"""
    with fits.open(input) as hdul:
        header = hdul[1].header
        data = hdul[1].header
        wcs = WCS(header, key="A")
        ccor_map = smap.Map(data, header, key="A")
        obs_time = header["DATE-OBS"]
        time = ts.from_astropy(Time(obs_time))

    return (data, wcs, ccor_map, time, obs_time)


def write_output(
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
):
    for i in range(len(date_obs)):
        file_name = os.path.basename(inputs[i])
        date = file_name.split("_")
        file_tstamp = "_".join([date[3], date[4]])
        creation = datetime.datetime.now().strftime("p%Y%m%dT%H%M%SZ")

        artifact_df = {}
        artifact_df["parent_file"] = file_name
        artifact_df["info"] = (
            "Values are in units pixels and reflect the locations at the L3 product level - "
            + "must scale locations by a factor of 2 for plotting over L1a-L2 products."
        )
        artifact_df["obs_time"] = date_obs[i]
        artifact_df["comet_name"] = all_comets[i][0] if len(all_comets[i]) > 0 else None
        artifact_df["comet_pos_x"] = all_valid_pixels[i][0][0] if len(all_valid_pixels[i]) > 0 else None
        artifact_df["comet_pos_y"] = all_valid_pixels[i][0][1] if len(all_valid_pixels[i]) > 0 else None
        artifact_df["moon_pos_x"] = all_moon_pixels[i][0]
        artifact_df["moon_pos_y"] = all_moon_pixels[i][1]
        artifact_df["mercury_pos_x"] = all_merc_pixels[i][0]
        artifact_df["mercury_pos_y"] = all_merc_pixels[i][1]
        artifact_df["venus_pos_x"] = all_venus_pixels[i][0]
        artifact_df["venus_pos_y"] = all_venus_pixels[i][1]
        artifact_df["mars_pos_x"] = all_mars_pixels[i][0]
        artifact_df["mars_pos_y"] = all_mars_pixels[i][1]
        artifact_df["jupiter_pos_x"] = all_jupiter_pixels[i][0]
        artifact_df["jupiter_pos_y"] = all_jupiter_pixels[i][1]
        artifact_df["saturn_pos_x"] = all_saturn_pixels[i][0]
        artifact_df["saturn_pos_y"] = all_saturn_pixels[i][1]
        artifact_df["neptune_pos_x"] = all_neptune_pixels[i][0]
        artifact_df["neptune_pos_y"] = all_neptune_pixels[i][1]
        artifact_df["uranus_pos_x"] = all_uranus_pixels[i][0]
        artifact_df["uranus_pos_y"] = all_uranus_pixels[i][1]
        artifact_df["stars_x"] = all_stars[i][0].tolist()
        artifact_df["stars_y"] = all_stars[i][1].tolist()
        artifact_df["star_names"] = get_star_names[i]
        artifact_df["star_hip_id"] = all_star_ids[i]

        try:

            with open(
                "/Users/vicente.salinas/Desktop/CCOR_Testing/test_object_identification/ccor1-id/2024-10-27-comet/"
                + f"sci_ccor1-obj_g19_{file_tstamp}_{creation}_pub.json",
                "w",
            ) as data_file:
                json.dump(artifact_df, data_file, indent=4)
        except TypeError:
            print("No data to output...skipping file")
            pass
