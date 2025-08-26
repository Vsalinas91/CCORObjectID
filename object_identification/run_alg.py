import glob
import argparse
import ccor_id


def do_processing(input_dir: str, generate_figures: bool = False, write_output_files: bool = False) -> None:
    """
    Run this script to execute the object identification using
    data specified in the input_dir input argument on the command line.
    """
    # get files:
    files = sorted(glob.glob(input_dir + "*.fits"))[38:39]

    # execute algorithm for input file stack:
    # try:
    ccor_id.run_alg(files, generate_figures, write_output_files)
    # except Exception:
    #     print("Cannot process the current input file stack.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-f", "--gen_figures", type=bool, required=False)
    parser.add_argument("-w", "--write_outputs", type=bool, required=False)

    args = parser.parse_args()
    input_dir = args.input_dir
    generate_figures = args.gen_figures
    write_outputs = args.write_outputs

    do_processing(input_dir, generate_figures, write_outputs)
