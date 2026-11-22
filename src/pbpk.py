"""Minimal PBPK engine (v2): gut -> liver -> central blood <-> peripheral tissue.

Units: amounts in mg, volumes in L, flows in L/h, time in h, concentrations in mg/L.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Indices of the state vector y, so the code reads clearly.
GUT, LIV, CEN, PER, ELIM, LOST, AUC = range(7)


def load_params(drug, path=None):
    """Read data/drug_parameters.csv and return {parameter: value} for one drug."""
    path = Path(path) if path else DATA_DIR / "drug_parameters.csv"
    df = pd.read_csv(path)
    rows = df[df["drug"] == drug]
    if rows.empty:
        raise ValueError(f"No parameters found for drug '{drug}'")
    return dict(zip(rows["parameter"], rows["value"].astype(float)))


def allele_activity(allele, a17):
    """Activity of one CYP2C19 allele: no-function 0, normal 1, increased a17."""
    if allele in ("*2", "*3"):
        return 0.0
    if allele == "*1":
        return 1.0
    if allele == "*17":
        return float(a17)
    raise ValueError(f"Unknown allele {allele}")


def person_activity(allele1, allele2, a17):
    """Mean activity of a person's two alleles (e.g. '*1','*17')."""
    return 0.5 * (allele_activity(allele1, a17) + allele_activity(allele2, a17))


def clint_total(p, activity):
    """Total unbound CLint: CYP2C19 part scales with activity; other pathways fixed."""
    clint_2c19 = p["fm"] * p["CLint_NM"] * activity
    clint_other = (1.0 - p["fm"]) * p["CLint_NM"]
    return clint_2c19 + clint_other


REF_WEIGHT = 70.0  # kg; the parameter file describes a 70 kg reference adult


def simulate(p, activity=1.0, env=None, dose=None, t_end=72.0, weight_kg=REF_WEIGHT):
    """Solve the PBPK ODEs for one person. Returns dict of curve + summary metrics."""
    env = env or {}
    size = weight_kg / REF_WEIGHT   # 1.0 for the reference adult
    dose = p["dose_mg"] if dose is None else dose

    # Apply environment multipliers (all default to 1.0 = Earth).
    ka = p["ka"] * env.get("ka", 1.0)
    fa = min(p["fa"] * env.get("fa", 1.0), 1.0)
    Qh = p["Qh"] * size ** 0.75 * env.get("Qh", 1.0)
    V_cen = p["V_cen"] * size * env.get("V_cen", 1.0)
    V_liv = p["V_liv"] * size
    K_liv = p["K_liv"]
    fu = p["fu"] * env.get("fu", 1.0)
    CLint = clint_total(p, activity) * size ** 0.75 * env.get("CLint", 1.0)
    k_cp, k_pc = p["k_cp"], p["k_pc"]

    def rhs(t, y):
        c_cen = y[CEN] / V_cen
        c_liv_out = y[LIV] / V_liv / K_liv      # hepatic venous concentration
        elim_rate = fu * CLint * c_liv_out
        absorb = ka * y[GUT]
        return [
            -absorb,                                              # gut
            fa * absorb + Qh * (c_cen - c_liv_out) - elim_rate,   # liver
            Qh * (c_liv_out - c_cen) - k_cp * y[CEN] + k_pc * y[PER],  # central
            k_cp * y[CEN] - k_pc * y[PER],                        # peripheral
            elim_rate,                                            # eliminated (bookkeeping)
            (1.0 - fa) * absorb,                                  # never absorbed (bookkeeping)
            c_cen,                                                # running AUC
        ]

    # Dense grid early (to catch Cmax/tmax), coarser later.
    t_eval = np.unique(np.concatenate([np.linspace(0, 6, 1201), np.linspace(6, t_end, 800)]))
    y0 = [dose, 0, 0, 0, 0, 0, 0]
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval, method="LSODA", rtol=1e-9, atol=1e-12)

    conc = sol.y[CEN] / V_cen
    i_max = int(np.argmax(conc))
    return {
        "t": sol.t, "conc": conc, "states": sol.y,
        "AUC": sol.y[AUC][-1], "Cmax": conc[i_max], "tmax": sol.t[i_max],
        "dose": dose, "fa": fa, "fu": fu, "CLint": CLint, "Qh": Qh,
    }


def analytic(p, activity=1.0, env=None, weight_kg=REF_WEIGHT):
    """Closed-form properties from the equations (used as verification targets)."""
    env = env or {}
    size = weight_kg / REF_WEIGHT
    Qh = p["Qh"] * size ** 0.75 * env.get("Qh", 1.0)
    fu = p["fu"] * env.get("fu", 1.0)
    fa = min(p["fa"] * env.get("fa", 1.0), 1.0)
    CLint = clint_total(p, activity) * size ** 0.75 * env.get("CLint", 1.0)
    x = fu * CLint
    return {"CL_h": Qh * x / (Qh + x), "F_h": Qh / (Qh + x), "F": fa * Qh / (Qh + x),
            "AUC_per_dose": fa / x}


def terminal_half_life(res, t_from=8.0, t_to=24.0):
    """Terminal half-life from a log-linear fit of the late curve."""
    m = (res["t"] >= t_from) & (res["t"] <= t_to) & (res["conc"] > 0)
    slope = np.polyfit(res["t"][m], np.log(res["conc"][m]), 1)[0]
    return np.log(2) / -slope


if __name__ == "__main__":
    p = load_params("omeprazole")
    for name, act in [("PM", 0.0), ("NM", 1.0), ("UM (*17/*17)", p["a17"])]:
        r = simulate(p, act)
        print(f"{name:14s} AUC={r['AUC']:.3f} mg*h/L  Cmax={r['Cmax']:.3f} mg/L  "
              f"tmax={r['tmax']:.2f} h  t1/2={terminal_half_life(r):.2f} h")