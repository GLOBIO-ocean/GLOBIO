"""Constants shared by data handling, training and evaluation."""
import numpy as np

#: Target variables supported by the models.
VARIABLES = ("CHLA", "NITRATE", "BBP700")

#: Number of vertical levels of every profile (inputs and target).
N_LEVELS = 200

#: Maximum pressure (dbar) and vertical step (dbar) of the profile grid of each target.
MAX_PRESSURE = {"NITRATE": 1000, "CHLA": 200, "BBP700": 200}
PRESSURE_STEP = {"NITRATE": 5, "CHLA": 1, "BBP700": 1}

#: Physical units of the targets, used in reports and plots.
UNITS = {"NITRATE": "mmol m-3", "CHLA": "mg m-3", "BBP700": "1e-3 m-1"}

#: Reference year of the year encoding (the model sees ``year - YEAR_OFFSET``).
YEAR_OFFSET = 2010

#: Values >= FILL_VALUE are fill values (missing data) in the dataset CSVs.
FILL_VALUE = 99999.0

#: Argo global range test for the inputs: (low, high), inclusive.
INPUT_RANGES = {
    "temp": (-2.5, 40.0),
    "psal": (2.0, 41.0),
    "doxy": (-5.0, 600.0),
}

#: Argo global range test for the targets, used by ``tools/clean_dataset.py``.
TARGET_RANGES = {
    "NITRATE": (-2.0, 50.0),
    "CHLA": (-0.1, 100.0),
    "BBP700": (-0.000025, 0.1),
}

#: Row labels of the dataset CSVs; the eighth row holds the target and is labelled
#: with the variable name (e.g. ``CHLA``).
METADATA_ROWS = ("year", "day_rad", "lat", "lon")
INPUT_ROWS = ("temp", "psal", "doxy")

#: Geographical boxes. ``lat`` is (min, max); ``lon`` is (min, max) or a list of
#: such ranges. Bounds are inclusive. ``r1``-``r6`` are the 30-degree latitude
#: bands used for the global-model analysis.
REGIONS = {
    "mediterranean": {"lat": (30, 46), "lon": (-6, 36)},
    "indian_ocean": {"lat": (-40, 30), "lon": (20, 120)},
    "bay_of_bengal": {"lat": (8, 22), "lon": (82, 98)},
    "north_atlantic": {"lat": (45, 70), "lon": (-70, 10)},
    "southern_ocean": {"lat": (-90, -60), "lon": (-180, 180)},
    "north_pacific": {"lat": (45, 65), "lon": (-180, -120)},
    "pacific": {"lat": (-60, 65), "lon": [(120, 180), (-180, -70)]},
    "r1": {"lat": (60, 90), "lon": (-180, 180)},
    "r2": {"lat": (30, 60), "lon": (-180, 180)},
    "r3": {"lat": (0, 30), "lon": (-180, 180)},
    "r4": {"lat": (-30, 0), "lon": (-180, 180)},
    "r5": {"lat": (-60, -30), "lon": (-180, 180)},
    "r6": {"lat": (-90, -60), "lon": (-180, 180)},
}
LATITUDE_BANDS = ("r1", "r2", "r3", "r4", "r5", "r6")
BASINS = tuple(name for name in REGIONS if name not in LATITUDE_BANDS)

#: Indices of the CHLA grid (1 dbar step) closest to the 22 depth levels of the
#: SOCA product, used when comparing against SOCA.
SOCA_LEVEL_INDICES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
                      80, 90, 100, 125, 150, 175, 199]


def check_variable(variable):
    """Return ``variable`` in upper case, raising if it is not supported."""
    name = str(variable).upper()
    if name not in VARIABLES:
        raise ValueError(f"Unknown variable {variable!r}; expected one of {VARIABLES}")
    return name


def pressure_levels(variable):
    """Pressure (dbar) of the 200 levels of the profiles of ``variable``."""
    variable = check_variable(variable)
    levels = np.arange(0, MAX_PRESSURE[variable], PRESSURE_STEP[variable], dtype=float)
    assert levels.size == N_LEVELS
    return levels
