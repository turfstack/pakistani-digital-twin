"""Add sex, age, weight and a group column to a cohort file; then run the PBPK engine on it."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pbpk import load_params, person_activity, simulate  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42

# PLACEHOLDER adult body-weight assumptions (kg). Replace with a cited Pakistani survey.
WEIGHT_MEAN = {"male": 72.0, "female": 62.0}
WEIGHT_SD = {"male": 11.0, "female": 10.0}


def add_covariates(cohort, seed=SEED):
    """Return a copy of the cohort with sex, age_years, weight_kg and ethnic_group columns."""
    rng = np.random.default_rng(seed)
    out = cohort.copy()
    n = len(out)

    out["sex"] = np.where(rng.random(n) < 0.5, "male", "female")
    out["age_years"] = np.clip(rng.normal(38, 9, n), 25, 60).round(1)

    mean = out["sex"].map(WEIGHT_MEAN).to_numpy()
    sd = out["sex"].map(WEIGHT_SD).to_numpy()
    out["weight_kg"] = np.clip(rng.normal(mean, sd), 40, 130).round(1)

    out["ethnic_group"] = "pooled"
    return out


def add_activity(cohort, a17):
    """Add each person's mean CYP2C19 allele activity from allele_1 and allele_2."""
    out = cohort.copy()
    act = {"*1": 1.0, "*2": 0.0, "*3": 0.0, "*17": float(a17)}
    out["activity"] = 0.5 * (out["allele_1"].map(act) + out["allele_2"].map(act))
    return out


def run_sample(cohort, drug="omeprazole", n=2000, seed=SEED):
    """Simulate a random sample of people and return their exposure metrics."""
    p = load_params(drug)
    cohort = add_activity(cohort, p["a17"])
    sample = cohort.sample(n=min(n, len(cohort)), random_state=seed)
    rows = []
    for r in sample.itertuples():
        res = simulate(p, activity=r.activity, weight_kg=r.weight_kg, t_end=48.0)
        rows.append((r.person_id, res["AUC"], res["Cmax"], res["tmax"]))
    metrics = pd.DataFrame(rows, columns=["person_id", "AUC", "Cmax", "tmax"])
    return sample.merge(metrics, on="person_id")


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "pk2021"
    src = DATA_DIR / f"cohort_cyp2c19_{scenario}.csv"
    cohort = add_covariates(pd.read_csv(src))
    dst = DATA_DIR / f"cohort_cyp2c19_{scenario}_cov.csv"
    cohort.to_csv(dst, index=False)
    print("Saved", dst, "-", len(cohort), "people")

    result = run_sample(cohort)
    print(result.groupby("phenotype")[["AUC", "Cmax", "tmax"]].median().round(3))