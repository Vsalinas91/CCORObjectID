# Satellite Identification
Satellite identification requires a different workflow that identfying and cataloguing celestial bodies. The data source(s) for determining the position of a satellite relative to the `DATE` timestamps of a FITs image product either are limited to current (most up-to-date) satellite ephemeris data, or data that have been archived and require an account login for access. These data come in the form of two-line element (TLE) data files are are used to determine the orbital position of satellites relative to an observatory at the times of image capture. Below is a list of TLE data sources:

 1. CelesTrak: current satellite TLE data files. Does not require an account, but only contains data that are current, and valid within a 7 day period. Outside of this time period, orbital predictions become unreliable. (https://celestrak.org/NORAD/elements/)
 2. Space-Track: archived satellite TLE data files. Requires an account for downloading/requesting these archived data. Archived TLE's require some pre-processing included in the code found in `sat_id.py` where duplicate entries for select satellites may be present: older and corrected data. (https://www.space-track.org)

There is no guarantee that use of temporally relavent TLE data files will allow for correct satellite identification within an observatory's image data. However, the methodology implemented in this repository is simple and straight forward where issues may be present with coordinate transformations, the type of world coordinate system/reference coordinate frame used, or simply with bad TLE data.


# Methodology: 
The method to identify satellites requires only the 3D positions of satellite data, and the 3D position of the observatory retrieved from image metadata/ephemeris data:

 1. Ingest TLE for relevant image date,
 2. Extract observer coordinates for image data,
 3. Determine satellite positions for image time,
 4. Cast observer and satellite positions into identical coordinate system - in this code Geocentric Earth Equatorial is used.
 5. Determine Observatory-Sun line-of-sight vector, analagous to instrument's boresight (heliocentric),
 6. For each satellite, calculate the angle (`sat_angle`) from the observatory to it's position relative to the boresight,
 7. Determine candidate satellites *if* it's `sat_angle` is within the instrument's field-of-view (FOV).

If candidate satellites are found, the `sat_angle` may be used to plot a general location relative to the image center. An additional method is included in `find_satellites.py` that enable the approximate of the satellite angular coordinates relative to the image center as both azimuth and inclination angles from the image's bottom. These data are saved to a `.json` file from which the positions may be used to overlay onto the instrument's imagery to validate it's position to the actual image artifact (see examples below).