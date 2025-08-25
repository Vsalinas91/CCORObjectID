import json
import os
import datetime

from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from sunpy import map as smap
from pathlib import Path

from dataclasses import dataclass
from typing import Any
import numpy.typing as npt
from skyfield.timelib import Timescale

from .exceptions import CCORExitError

ROOT_DIR = Path(__file__).parent.parent


@dataclass(frozen=True, kw_only=True)
class GetData:
    image_data: npt.NDArray[Any]
    header: fits.Header
    WCS: WCS
    ccor_map: smap.GenericMap
    time: Timescale
    obs_time: str
    end_time: str


def read_input(input: str, ts: Timescale) -> GetData:
    """Read fits file and return data and header"""
    with fits.open(input) as hdul:
        header = hdul[1].header
        data = hdul[1].data
        wcs = WCS(header, key="A")
        ccor_map = smap.Map(data, header, key="A")
        obs_time = header["DATE-OBS"]
        end_time = header["DATE-END"]
        time = ts.from_astropy(Time(obs_time))

    return GetData(
        image_data=data, header=header, WCS=wcs, ccor_map=ccor_map, time=time, obs_time=obs_time, end_time=end_time
    )


def write_output(obs_time: str, end_time: str, data_dict: dict[str, Any]) -> None:
    """
    write out the data to a file that matches the data product cadence timestamp.
    """
    obs_time_fmt = obs_time.replace("-", "").replace(":", "").split(".")[0]
    end_time_fmt = end_time.replace("-", "").replace(":", "").split(".")[0]
    file_tstamp = f"s{obs_time_fmt}Z_e{end_time_fmt}Z"
    out_dir = f"{obs_time_fmt.split('T')[0]}"
    creation = datetime.datetime.now().strftime("p%Y%m%dT%H%M%SZ")

    # Create output directory if it does  not exist:
    try:
        os.makedirs(os.path.join(ROOT_DIR, f"outputs/{out_dir}"), exist_ok=True)
    except OSError as e:
        print(f"Error creating data directory: {str(e)}")

    try:
        with open(
            os.path.join(ROOT_DIR, f"outputs/{out_dir}/sci_ccor1-obj_g19_{file_tstamp}_{creation}_pub.json"),
            "w",
        ) as data_file:
            json.dump(data_dict, data_file, indent=4)
    except TypeError:
        raise CCORExitError("No data to output...skipping file")


def get_vignetting_func() -> npt.NDArray[Any]:
    """
    Retrieve the vignetting function for plotting.
    """
    with fits.open(os.path.join(ROOT_DIR, "static_required/vig_ccor1_20250209.fits")) as hdul:
        return hdul[0].data
