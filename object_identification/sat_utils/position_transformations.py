import numpy as np
from dataclasses import dataclass
import numpy.typing as npt
from typing import Any


@dataclass(frozen=True, kw_only=True)
class GetSatelliteProjection:
    xproj: float
    yproj: float


@dataclass(frozen=True, kw_only=True)
class GetSatelliteHelioprojective:
    Tx: float
    Ty: float


def get_sat_angle(sat: npt.NDArray[Any], ccor: npt.NDArray[Any], sun: npt.NDArray[Any]) -> float:
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


def get_sat_az(x_proj: float, y_proj: float) -> float:
    """
    Calculate the azimuth angle of the projected satellite image on the observer plane.
    The coordinates x_proj and y_proj are the satellite's projection location centered on the sun.

                                .  |  .   Y (SOLAR NORTH)
                             .     |    .
                           o       |      .
                            \      |       .                                   # noqa: W605
                               \   |        .                                  # noqa: W605
                          --------- ------------ X (WEST LIMB)
                                   |
                                   |
                                   |
                                   |

    Args:
       x_proj = projected x coordinate of satellite
       y_proj = projected y coordinate of satellite

    Returns:
       Azimuthal angle of the satellite position on the observer 2D plane
       from solar west.

    """
    # Calculate the azimuthal angle
    azimuth_rad = np.arctan2(y_proj, x_proj)
    return np.degrees(azimuth_rad)


def do_sat_projection(obs_coords: npt.NDArray[Any], sat_coords: npt.NDArray[Any], sun_coords: npt.NDArray[Any]):
    """
    Calculates the projected satellite position on a 2D observer plane
    orthogonal to the observer-sun vector. For solar imagers, the
    coordinates provided should be in a heliocentric cartesian coordinate
    system such as Helioprojective.

    This method assumes that z = Solar North allowing for the projected locations to
    be correctly aligned with the image (after rotation with crota).

    Args:
        obs_coords (np.array): 3D position vector of the observer.
        sun_coords (np.array): 3D position vector of the Sun.
        sat_coords (np.array): 3D position vector of the satellite.

    Returns:
        x_proj-sun_x_proj = sun centered satellite projection x coordinate
        y_proj-sun_y_proj = sun centered satellite projection y coordinate
    """
    # Define vectors and local coordinate system
    # Vector from observer to sun
    G = sun_coords - obs_coords
    Gn = G / np.linalg.norm(G)

    # Celestial North vector in ICRS frame
    z = np.array([0, 0, 1])

    # Local Y-axis (North direction in image plane)
    y_frame = z - np.dot(z, Gn) * Gn
    Un = y_frame / np.linalg.norm(y_frame)

    # Local X-axis (East direction in image plane)
    Rn = np.cross(Un, Gn)

    # Project satellite vector onto the plane
    # Vector from observer to satellite
    S = sat_coords - obs_coords

    x_proj = np.dot(S, Rn)
    y_proj = np.dot(S, Un)

    sun_x_proj = np.dot(G, Rn)
    sun_y_proj = np.dot(G, Un)

    return GetSatelliteProjection(
        xproj=x_proj - sun_x_proj,
        yproj=y_proj - sun_y_proj,
    )


def angle_to_helioprojective(sat_angle: float, sat_az: float) -> GetSatelliteHelioprojective:
    """
    Using the sat_angle and sat_az angles, calculate the approxite helioprojective
    position in units of degrees from the observer's boresight.

    Note: sat_angle in degrees
          sat_az in radians.
    """
    Tx = (sat_angle) * np.sin(sat_az)
    Ty = (sat_angle) * np.cos(sat_az)

    return GetSatelliteHelioprojective(
        Tx=Tx,
        Ty=Ty,
    )


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
