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
