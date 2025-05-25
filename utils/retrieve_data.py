from skyfield.api import load
from skyfield.data import hipparcos, mpc, stellarium


def load_planetary_data():
    """
    Load in planetary ephemeris data.
    """
    ephemeris = load("de421.bsp")
    return (ephemeris["earth"], ephemeris["sun"])


def load_star_data():
    """
    Load the Hipparcos start catalogue data.
    """
    try:
        with load.open(hipparcos.URL) as f:
            return hipparcos.load_dataframe(f)
    except Exception:
        print("Could not load star catalogue - try different source.")
        return None


def load_comet_data():
    """
    Load the comet data.
    """
    try:
        with load.open(mpc.COMET_URL) as f:
            comets = mpc.load_comets_dataframe(f)
            # Resort the dataframe:
            return (
                comets.sort_values("reference")
                .groupby("designation", as_index=False)
                .last()
                .set_index("designation", drop=False)
            )
    except Exception:
        print("Could not load comet data - try different source.")
        return None


def load_constellation_data():
    """
    Load the constellation data - major constellations.
    """
    const_url = (
        "/Users/vicente.salinas/Desktop/CCORObjectID/ccor_object_identification/static_required/constellationship.fab"
    )
    with load.open(const_url) as f:
        return stellarium.parse_constellations(f)
