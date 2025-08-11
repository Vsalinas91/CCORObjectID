# CCOR Object Identification
CCOR imagery are captured in visible white light which means that celestial bodies and objects that transit through the instrument's field-of-view (FOV) are visible in the image data. Such objects include planets, the moon, comets, and even satellites since CCOR-1 is in geostationary orbit on-board GOES-19.

The object identification routines found in this repository allow for the successful identification of celestial objects in any valid CCOR-1 image data file, namely level 3 downsampled data due to smaller file sizes using various ephemeris data sources all handled by the `SkyField` python package which is included as a dependency in the `environment.yml` file. 

Object identification relies on valid CCOR product metadata, namely the observation time of the image capture. The include ephemeris data are used to determine the location(s) of celestial bodies/objects at the observation time and relative the CCOR world coordinate system (WCS).

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


