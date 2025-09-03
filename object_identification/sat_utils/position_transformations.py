import numpy as np


def get_angle(sat, ccor, sun):
    """
    Get the angle between two vectors (2d or 3d), this angle represents the angle
    for two given vectors in 3D


            |
            |       SAT
            |      x x x x x x SUN
            |     x           x
            |    x         x
            |   x  O    x
            |  x     x
            | x   x
            |x x
            ------------------------

    """
    # Example points A, B, C in 3D
    A = sun
    B = ccor
    C = sat

    # Vectors from B to A and C
    BA = A - B
    BC = C - B

    # Compute angle at B in radians
    cos_theta = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    theta_rad = np.arccos(cos_theta)

    # Optionally convert to degrees
    theta_deg = np.degrees(theta_rad)

    # print(f"Angle at vertex B: {theta_rad:.4f} radians, or {theta_deg:.2f} degrees")
    return theta_deg


def get_angular_positions(sat, ccor, sun):
    """
    Assuming the satellite angle is a vector magnitude in
    coordinates in degrees, find the x and y angular components
    (i.e., azimuth and inclination).

    To do this, we have to be creative: First, calculate the angle between the
    CCOR-SUN FOV and the CCOR-SAT Locations in 2D, so the 2D Projection of the angles
    (see get_angle) docstring.

    Then, calculate the angle from the x-axis to the Sun and the sattelite, this will
    help us determine if the satellite is to the right or left of the boresight

            |
            |       SAT
            |      x           SUN      O=angle from x to SAT
            |     x           x         A=angle from x to SUN
            |    x         x
            |   x  O    x               FACTOR=-1 if O>A (LEFT of boresight) else 1 if O<A (RIGHT of boresight)
            |  x     x
            | x   x
            |x x      A
            ------------------------


    Then, using the 3D vector positions, calculate the satellite and sun elevation angles. The difference
    between SAT elevation and Sun elevation will give us the approximate position of the
    y-angular component that falls on the circle. This assumes that we project the elevation angles
    onto a 2D vertical plane and the difference between the two angles gives us the approximate angle
    between the CCOR-SUN line-of-sight vector and the satellite position.

    """
    # First, get the azimuthal angle relative to the x-y plane only
    azim = get_angle(sat[:2], ccor[:2], sun[:2])

    # Now, determine if the satellite is left or right of boresight
    angle_to_sun_horizontal = np.rad2deg(np.arctan2((sun[1] - ccor[1]), (sun[0] - ccor[0])))
    angle_to_sat_horizontal = np.rad2deg(np.arctan2((sat[1] - ccor[1]), (sat[0] - ccor[0])))
    if angle_to_sun_horizontal < angle_to_sat_horizontal:
        factor = -1
    else:
        factor = 1

    # Redefine the vector magnitudes
    spos = ((sat[0] - ccor[0]) ** 2 + (sat[1] - ccor[1]) ** 2 + (sat[2] - ccor[2]) ** 2) ** 0.5
    pos = ((sun[0] - ccor[0]) ** 2 + (sun[1] - ccor[1]) ** 2 + (sun[2] - ccor[2]) ** 2) ** 0.5

    # Calculate the vertical angle from the horizontal plane from the origin (ccor)
    sat_el = np.rad2deg(np.arccos((sat[-1] - ccor[-1]) / spos))

    # Now, calculate the vertical angle from ccor to the sun and subtract the sat vertical angle
    # to get the angle between them
    sun_ccor_angle = np.rad2deg(np.arccos((sun[-1] - ccor[-1]) / pos))
    s_inclination = sat_el - sun_ccor_angle

    return (factor * azim, s_inclination)


def get_pixel_locations(sat, ccor, sun, yaw_status, ccor_fov=11.0):
    """
    Get approximate pixel locations of projected satellite vectors on CCOR Plane
    """
    # Example points A, B, C in 3D
    A = sun
    B = ccor
    C = sat

    # Vectors from B to A and C
    F = A - B  # Sun-CCOR
    V = C - B  # Sat-CCOR

    # Define unit normal of the SUN-CCOR and SAT-CCOR Vectors
    F_norm = F / np.linalg.norm(F)
    V_norm = V / np.linalg.norm(V)

    # Define World Up Vector along z axis
    U_world = np.array([-1, -1, 1])  # use the [0,0,1] to get the location relative to our aperature
    # U_world = rotate_up_vector(U_world, F_norm, np.deg2rad(header['CROTA']))

    # Define the image radius
    # For level 3 images, need to divide by 2
    image_radius = 5.5 * 3600 / 19.30 / 2  # noqa: F841

    # Define the "image" Right Vector or vector
    # defining the "horizontal" axis
    R = np.cross(U_world, F_norm)
    if np.linalg.norm(R) < 1e-6:
        print("update world")
        U_world = np.array([0, 1, 0])
        R = np.cross(U_world, F_norm)
    R_norm = R / np.linalg.norm(R)

    # Calculate the true up-vector that defines the vertical
    # image axis:
    U_img = np.cross(F_norm, R_norm)
    U_norm = U_img / np.linalg.norm(U_img)

    # Now, project satellite vector to image plane:
    V_perp = V_norm - np.dot(np.dot(V_norm, F_norm), F_norm)
    # V_perp = rotate_up_vector(U_world, F_norm, np.deg2rad(26))

    # Get pix projections onto the orthonormal axes:
    x = np.dot(V_perp, R_norm)
    y = np.dot(V_perp, U_norm)
    z = np.dot(V_perp, F_norm)  # noqa: F841

    # Now, we want to define the scaling parameters for getting the
    # x and y locations into units pixels
    width = 1024  # 1024
    height = 960  # 960

    fov_hor = np.deg2rad(ccor_fov)
    pix_per_radx = width / fov_hor

    aspect_ratio = width / height  # noqa: F841
    fov_vert = np.deg2rad(ccor_fov)  # 2 * np.arctan(np.tan(fov_hor / 2) * aspect_ratio)
    pix_per_rady = height / fov_vert

    pix_x = width / 2 + x * pix_per_radx  # use width/2 instead of header['CRPIX1'] if desired
    pix_y = height / 2 - y * pix_per_rady

    # new:
    # pix_x = (x/z) * (width/2) / np.arctan(fov_hor/2) + width/2
    # pix_y = -(y/z) * (height/2) / np.arctan(fov_hor/2) + height/2

    if yaw_status > 0:
        return (
            width - pix_x,
            height - pix_y,
        )  # this will depend on the yaw state, so we'll need to add a yaw_flip arg and not subtract if yaw_flip is 0
    else:
        return (pix_x, pix_y)


def rotate_point(point, center, angle_degrees):
    """Rotates a point around a center point by a given angle.

    Args:
        point (tuple): The (x, y) coordinates of the point to rotate.
        center (tuple): The (x, y) coordinates of the center of rotation.
        angle_degrees (float): The angle of rotation in degrees.

    Returns:
        tuple: The (x, y) coordinates of the rotated point.
    """
    x, y = point
    cx, cy = center
    angle_radians = np.deg2rad(angle_degrees)

    # Translate to origin
    translated_x = x - cx
    translated_y = y - cy

    # Rotate
    rotated_x = translated_x * np.cos(angle_radians) - translated_y * np.sin(angle_radians)
    rotated_y = translated_x * np.sin(angle_radians) + translated_y * np.cos(angle_radians)

    # Translate back
    final_x = rotated_x + cx
    final_y = rotated_y + cy

    return final_x, final_y
