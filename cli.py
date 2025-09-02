import logging
import argparse
import os
import glob

from object_identification.ccor_id import run_alg as run_ccor_id
from object_identification.sat_id import run_satellite_id

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to input data")
    parser.add_argument("-f", "--gen_figures", action="store_true", required=False, help="Generate Figure(s)")
    parser.add_argument(
        "-w", "--write_outputs", action="store_true", required=False, help="Write locations to output file"
    )
    parser.add_argument("-tle", "--tle_input", type=str, required=False, help="Path to TLE file")
    parser.add_argument(
        "--search_radius", type=float, required=False, help="Search radius for satellite near the observatory"
    )
    parser.add_argument("--fov_angle", type=float, required=False, help="Entire instrument FOV")

    args = parser.parse_args()
    input_dir = args.input_dir
    generate_figures = args.gen_figures
    write_outputs = args.write_outputs

    # for satellite id
    tle = args.tle_input
    search_radius = args.search_radius
    fov_angle = args.fov_angle

    # Stack Images:
    if os.path.exists(input_dir):
        files = sorted(glob.glob(input_dir + "*.fits"))[:]
        if tle is None:
            logger.info(f"Running object identification for: {len(files)} image frames.")
            # Run the alg
            run_ccor_id(files, generate_figures, write_outputs)
        else:
            logger.info(f"Running satellite identification for: {len(files)} image frames.")
            run_satellite_id(
                inputs=files,
                tle=tle,
                search_radius=search_radius,
                fov_angle=fov_angle,
                write_output_files=write_outputs,
            )
    else:
        logger.error("Invalid path to data...exiting")
