import logging
import argparse
import os
import glob

from object_identification.ccor_id import run_alg as run_ccor_id

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-f", "--gen_figures", type=bool, required=False)
    parser.add_argument("-w", "--write_outputs", type=bool, required=False)

    args = parser.parse_args()
    input_dir = args.input_dir
    generate_figures = args.gen_figures
    write_outputs = args.write_outputs

    # Stack Images:
    if os.path.exists(input_dir):
        files = sorted(glob.glob(input_dir + "*.fits"))[:1]
        logger.info(f"Running object identification for: {len(files)} image frames.")
        # Run the alg
        run_ccor_id(files, generate_figures, write_outputs)
    else:
        logger.error("Invalid path to data...exiting")
