"""Verification tests for the PBPK engine (plan section 5.8). Run: python tests/test_pbpk.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pbpk import analytic, load_params, simulate, terminal_half_life  # noqa: E402

p = load_params("omeprazole")


def test_mass_balance():
    r = simulate(p, 1.0)
    total = r["states"][:6].sum(axis=0)          # gut+liver+central+peripheral+eliminated+lost
    assert np.allclose(total, r["dose"], rtol=1e-6), "dose not conserved"


def test_numerical_auc_matches_analytic():
    for act in (0.0, 0.5, 1.0, p["a17"]):
        r = simulate(p, act)
        expected = analytic(p, act)["AUC_per_dose"] * r["dose"]
        assert abs(r["AUC"] - expected) / expected < 1e-3, (act, r["AUC"], expected)


def test_limit_zero_clearance():
    q = dict(p, CLint_NM=0.0)
    r = simulate(q, 1.0)
    assert r["states"][4][-1] < 1e-9, "something was eliminated with CLint = 0"


def test_limit_instant_absorption():
    r = simulate(p, 1.0, env={"ka": 1000.0})
    assert r["tmax"] < 0.1, "huge ka should give near-instant liver input"


def test_auc_independent_of_flow_but_cmax_not():
    base = simulate(p, 1.0)
    fast = simulate(p, 1.0, env={"Qh": 1.5})
    assert abs(fast["AUC"] - base["AUC"]) / base["AUC"] < 1e-3
    assert abs(fast["Cmax"] - base["Cmax"]) / base["Cmax"] > 0.01


def test_earth_nm_ranges():
    r = simulate(p, 1.0)
    assert 0.30 <= analytic(p, 1.0)["F"] <= 0.45, "bioavailability outside label-type range"
    assert 0.5 <= r["tmax"] <= 3.5, "tmax outside label-type range"
    assert 0.5 <= terminal_half_life(r) <= 1.5, "half-life outside label-type range"


def test_pm_em_ratio():
    ratio = simulate(p, 0.0)["AUC"] / simulate(p, 1.0)["AUC"]
    assert 3.0 <= ratio <= 6.6, ratio


def test_star17_ratio():
    ratio = simulate(p, 1.0)["AUC"] / simulate(p, p["a17"])["AUC"]
    assert abs(ratio - 2.1) < 0.1, ratio


def test_weight_scaling():
    base = simulate(p, 1.0)
    heavy = simulate(p, 1.0, weight_kg=105.0)
    expected = (105.0 / 70.0) ** -0.75           # AUC ~ 1/CLint ~ weight^-0.75
    assert abs(heavy["AUC"] / base["AUC"] - expected) < 1e-3
    assert heavy["Cmax"] < base["Cmax"], "bigger volume should lower Cmax"


def test_weight_scaling():
    base = simulate(p, 1.0)
    heavy = simulate(p, 1.0, weight_kg=105.0)
    expected = (105.0 / 70.0) ** -0.75           # AUC ~ 1/CLint ~ weight^-0.75
    assert abs(heavy["AUC"] / base["AUC"] - expected) < 1e-3
    assert heavy["Cmax"] < base["Cmax"], "bigger volume should lower Cmax"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"All {len(tests)} tests passed")