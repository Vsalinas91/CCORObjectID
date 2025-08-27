# CCOR Object Identification
CCOR imagery often contains artifcats that may correspond to celestial bodies and objects that transit through the its field-of-view (FOV). Such objects include planets, the moon, comets, and even satellites, since CCOR-1 is at geostationary orbit on-board GOES-19.

This object identification algorithm allows for identifying known celestial objects in any valid CCOR-1 image data file, namely level 3 downsampled data due to smaller file sizes. The algorithm uses the `SkyField` package to ingest ephemeris data from various sources listed below:

 1. Celestrak (or Space-Track for archive) - Earth satellite two-line element (TLE) objects.
 2. Minor Planet Center (MPC) - For comet orbital ephemeris data (may also use for obtaining astroids).
 3. JPL Horizons - planet/spacecraft SPICE Kernels.

These data, along with the CCOR ephemeris contained in the product metadata, are used to identify locations of objects within the image data based entirely off the observing time of the image capture.

Note: Object identification relies on valid CCOR product metadata, namely the observation time of the image capture. The include ephemeris data are used to determine the location(s) of celestial bodies/objects at the observation time and relative the CCOR world coordinate system (WCS).

# Current Identifiable Objects
The table below summarizes the objects that can currently be identified by the algorithm, and their locations (in units pixels) written to output files and overlaid onto the CCOR imagery for reference (or projected onto a reference object map).

| Object  | Identification |
| ------------- |:-------------:|
| Moon      | :heavy_check_mark:|
| Mercury   | :heavy_check_mark:|
| Venus     | :heavy_check_mark:|
| Mars.     | :heavy_check_mark:|
| Jupiter   | :heavy_check_mark:|
| Saturn    | :heavy_check_mark:|
| Neptune   | :heavy_check_mark:|
| Uranus    | :heavy_check_mark:|
| Major Constellations |:heavy_check_mark:|
| Major Stars| :heavy_check_mark:|
| Comets    |:heavy_check_mark:|
| Satellites| :x:              |

## Running the algorithm
To the run the algorithm:

> python -m cli -i /path/to/ccor1-l3/data/ -f -w

Where: 
 * -i (--input_dir): input directory
 * -f (--gen_figures): boolean to generate figures (True if set)
 * -w (--write_outputs): boolean for generating output file containing object coordinates (True if set)

 Note: the CCOR vignetting file needs to be placed in the `static_required` directory as it is not available in this repository.


## Future Work

This code is still under active delopement with the following items as potential future tasks/updates or additions:

 * Use SunPy and CCOR Spice Kernel for planet and moon location identification
 * Identify astroids using the Minor Planetary Catalogue
 * Identify satellites/satellite streaks in image data
 * Provide image/animation examples
 * Determine importance of location accuracy for this as a reference product
 * More TBD

## Examples: 
Constellation(s):
<img width="1660" height="644" alt="ccor1-l3-frame-078" src="https://github.com/user-attachments/assets/ce221abc-f26d-4cb5-916e-516b4d6d0723" />

Planet(s) and Moon:
<img width="1660" height="644" alt="ccor1-l3-frame-042" src="https://github.com/user-attachments/assets/f03eab4b-4653-4780-ac9e-835a5ed73851" />


