"""Earth / space / Mars environment multipliers for the PBPK engine."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GRAVITY = {"earth": 1.0, "space": 0.0, "mars": 0.38}   # in g


def load_environment(name, rng=None, path=None):
    """Return {parameter: multiplier} for 'earth', 'space' or 'mars'.

    rng=None uses the central multiplier; with a numpy Generator each multiplier is
    drawn uniformly between its low and high values (uncertainty sampling).
    """
    name = name.lower()
    if name not in GRAVITY:
        raise ValueError(f"Unknown environment '{name}'")
    if name == "earth":
        return {}                                   # all multipliers 1.0

    path = Path(path) if path else DATA_DIR / "environments.csv"
    rows = pd.read_csv(path)
    rows = rows[rows["environment"] == "space"]     # the 0 g effect sizes

    # Mars: assumed linear between 1 g (no effect) and 0 g (full space effect).
    strength = 1.0 - GRAVITY[name]                  # space 1.0, Mars 0.62

    env = {}
    for r in rows.itertuples():
        m = r.multiplier if rng is None else rng.uniform(r.low, r.high)
        env[r.parameter] = 1.0 + (m - 1.0) * strength
    return env