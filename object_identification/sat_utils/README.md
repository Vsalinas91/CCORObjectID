# Satellite Identification
Satellite identification requires a different workflow than identfying and cataloguing celestial bodies. The data source(s) for determining the position of a satellite relative to observing period of a FITs image product either are limited to current (most up-to-date) satellite ephemeris data, or data that have been archived and require an account login for access. These data come in the form of two-line element (TLE) data files and are used to determine/project the orbital position of satellites relative to an observatory at the times of image capture. Below is a list of TLE data sources:

 1. CelesTrak: current satellite TLE data files. Does not require an account, but only contains data that are current, and valid within a 7 day period. Outside of this time period, orbital predictions become unreliable. (https://celestrak.org/NORAD/elements/)
 2. Space-Track: archived satellite TLE data files. Requires an account for downloading/requesting these archived data. Archived TLE's require some pre-processing included in the code found in `sat_id.py` where duplicate entries for select satellites may be present: older and corrected data. (https://www.space-track.org)

There is no guarantee that use of temporally relavent TLE data files will allow for correct satellite identification within an observatory's image data. However, the methodology implemented in this repository attempts to identify satellite streaks for a larger time priod surrounding the observing time of the data product in order to determine an approximate position relative to the signal seen in the image data.

# Methodology: 
The method to identify satellites requires only the 3D positions of sun, satellite(s), and the observatory retrieved from image metadata and/or ephemeris data:

 1. Ingest TLE for relevant image date,
 2. Extract observer coordinates for image data *or* from the TLE (recommended if not using satellite spice kernel),
 3. Define a sequence of times by which to propagate the satellite's position surrounding the start of the observing period. This is to ensure we can at least build a streak of satellite candidates in the field of view,
 4. Cast observer and satellite positions into identical coordinate system - in this code Geocentric Earth Equatorial is used.
 5. Determine Observatory-Sun line-of-sight vector, analagous to instrument's boresight (heliocentric),
 6. For each satellite, calculate the angle (`sat_angle`) from the observatory to it's position relative to the boresight,
 7. Determine candidate satellites *if* it's `sat_angle` is within the instrument's field-of-view (FOV).
 8. Cast observer and satellite coordinates into a heliprojective frame (use the observer coordinate frame) and project the 3D satellite position vector onto a 2D observer plane. 
 9. Calculate the azimuthal angle of the 2D projected position on the observer plane centered at the sun (`sat_az`),
 10. Caculate the approximate helioprojective coordinates of the satellite position in units degrees,
 11. Convert the approximated helioprojective coordinates to pixel coordinates using the observer's world coordinate system.

A summary of the methods used is provided below for reference. 

## Angular Separation: Calculating `sat-angle`: 
First, the angular separation between the observer-sun and observer-satellite position vectors must be calculated. To do this we must ensure that all positions are in the same coordinate frame. A geocentric ICRS frame is chosen as both our instrument and satellites of interest are already in this coordinate frame. 

For CCOR, this position vector is already made available in the metadata, or FITs file header. Or, it can be calculated from the ephemeris data; doing this will ensure that GOES-19 and all other satellites are aligned in time since they come from the same data source. For a satellite, the position vector for the observation time is in a Geocentric Celestial Reference System (GCRS) already, and so no coordinate transformations need to be done. The Sun's position can easily be transformed into a GCRS coordinate system by using the function `get_body('sun', observation_time)`. This natively casts the Sun's position in an ICRS coordinate frame which can then be transformed into GCRS.

When all coordinates are in the GCRS system, the angle that defines the separation between the two vectors is calculated, where the Sun-observer vectors defines our line-of-sight (LOS), or image center: 

```math
\rm \vec{sat_{rel}} = \vec{sat_{GCRS}} - \vec{obs_{GCRS}}
```
```math
\rm \vec{sun_{rel}} = \vec{sun_{GCRS}} - \vec{obs_{GCRS}}
```

These two vectors define the positions relative to our observer, then, the angle of separation is calculated using a dot product and solving for $\rm \theta$

```math
\rm \vec{a} \cdot \vec{b} = \|\vec{a}\|\|\vec{b}\| cos\theta
```

and for our vectors,

```math
\rm \theta = cos^{-1} \left[ \frac{\vec{sat_{rel}} \cdot \vec{sun_{rel}}} {\|\vec{sat_{rel}}\|\|\vec{sun_{rel}}\| } \right]
```

The angle $\rm \theta$ is defined from the observer-Sun LOS, or from the center of the image. As a result, the angle is effectively the helioprojective radial angle from the center of the image to it's outer field-of-view (FOV).

## Azimuthal Angle Relative to Observer-Sun LOS: Calculating `sat-az`: 
To directly convert the satellite's position vector into the oberver's coordinate frame, once must ensure that the origin of the satellite's position matches that of the observer coordinate. This is is trivial, but no success in converting the coordinates correctly was made. Instead, the helioprojective coordinates are manually calculated using the angular components that define the `sat_angle` ($\rm \theta$). To do this, the 3D position vector of a satellite is projected onto a 2D observer plane that is orthogonal to the observer-sun LOS.

First, the GCRS coordinates are cast back into the observing coordinate frame (i.e., Helioprojective). This is done so that our projected vectors are oriented relative to solar north and can be overlaid onto an image whose rotation is defined by the `CROTA` header key. The vectors are transformed to their helioprojective representation and are defined as,

```math
\rm \vec{sat_{hpc-rel}} = \vec{sat_{hpc}} - \vec{obs_{hpc}}
```
```math
\rm \vec{sun_{hpc-rel}} = \vec{sun_{hpc}} - \vec{obs_{hpc}}
```

where `hpc` is the helioprojective cartesian representation. Then, the local (image local) coordinate system that is orthogonal to the LOS is defined. First, the image `z-axis` is defined as the axis of the LOS,

```math
\rm \vec{z_{image}} = \frac{sun_{hpc-rel}} {\|sun_{hpc-rel}\|}
```

which is a normalized vector defining the total distance from the observer to the sun. Then, we define our solar north vector to project our helioprojective coordinate onto the image plane,

```math
\rm \vec{n_{solar}} = [0, 0, 1]
```

where a value of 1 along `z` aligns with our defined `z-axis`. This ensures that our projection plane's y-axis is aligned with solar north. These vectors are used to calculate our image projection plane axis. For our image `y-axis`, or image North, we project our solar north vector onto the plane perpendicular to the LOS,

```math
\rm \vec{v_{north}} = \vec{n_{solar}} - (\vec{n_{solar}} - \vec{z_{image}})\vec{z_{image}}
```
```math
\rm \vec{y_{image}} = \frac{\vec{v_{north}}} {\| \vec{v_{north}} \|}
```

and our `x-axis` is defined as the image East direction (solar west limb) and is calculated by the cross product of the y and z image vectors,

```math
\rm \vec{x_{image}} = \vec{z_{image}} \times \vec{y_{image}}
```

The satellite's 3D helioprojective position vector can now be projected onto our new image axes,

```math
\rm \vec{x_{project}} = \vec{sat_{hpc}} \cdot \vec{x_{image}}
```
```math
\rm \vec{y_{project}} = \vec{sat_{hpc}} \cdot \vec{y_{image}}
```

Then, the azimuthal angle is calculated to determine the 2D components of $\rm \theta$ (or `sat_angle`) which then allows for their pixel locations to be found relative to the image. The azimuthal angle is calculated from the image center by,

```math
\rm \phi = tan^{-1}(\frac{y_{project}}{x_{project}})
```

As with $\rm \theta$, we will use this angle to calculate the helioprojective positions relative to the image in units of degrees.

## Calculating Heliprojective Coordinates and Pixel Locations
With the angular position of the satellite(s) calculated, the helioprojective coordinates are found and represent an angular separation from the image center,

```math
\rm Tx = \theta sin(\phi)
```
```math
\rm Ty = \theta cos(\phi)
```

noting that `sin` and `cos` are flipped to account for our project image orientation. These values can then be used to convert their locations into the observer's WCS world coordinates by invoking the WCS world to pixel converter. In python, this is:

```python
wcs = WCS(header)
hpc_coords = np.array([[Tx, Ty]]) * u.deg
sat_pix = wcs.all_world2pix(hpc_coords, 0)
```

where the second input param `0` is for image's whose origin as the lower-left corner, a value of `1` would be used for images whose origin is at the top left of the image data.

The above method, and calculated pixel locations of the satellites, allows for them to be overlaid onto the CCOR image. However, a one can expect these locations to not be exact due to timing differences between the ephemeris for each position vector, and errors can also be expected due to the simplification of this problem not wholly using astrophysical coordinate transformation techniques handled by SunPy. 

# Results: 
The satellite data are saved into a `.json` file which includes entries for each candidate satellite found in the FOV, their positions for each time checked/evaluated that are used to define a streak, their distances from the observer, the times used to project their positions relative to the observing period, and their position errors correponding to an angular offset of `sat_angle` used to determine potiential areal coverage of where the satellite could have been given known errors with synthesizing both observer and satellite positions relative to a single observing period. 

The attached images exemplify the identification of satellites in CCOR data. Note, the positions of the identified satellites lie along Earth's ecliptic plane, thus further demonstrating that the image artifacts seen are likely due to incident reflections of sun-light onto the satellite projected as a glare on the CCOR image data.

