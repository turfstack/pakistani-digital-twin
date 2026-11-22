"""Run the PBPK engine over sampled people x drugs x environments -> data/sim_runs.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environments import GRAVITY, load_environment  # noqa: E402
from pbpk import clint_total, load_params, simulate  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42
DRUGS = ["omeprazole", "acetaminophen"]
ENVS = ["earth", "space", "mars"]
ENV_PARAMS = ["ka", "fa", "V_cen", "Qh", "CLint", "fu"]
ALLELE_ACTIVITY = {"*1": 1.0, "*2": 0.0, "*3": 0.0}   # *17 handled with the sampled a17


def load_ranges(drug):
    """Return {parameter: (low, high)} for one drug from drug_parameters.csv."""
    df = pd.read_csv(DATA_DIR / "drug_parameters.csv")
    df = df[df["drug"] == drug]
    return {r.parameter: (r.low, r.high) for r in df.itertuples()}


def activity_of(a1, a2, a17):
    """Mean activity of two alleles, using the sampled a17 for *17."""
    act = lambda a: a17 if a == "*17" else ALLELE_ACTIVITY[a]
    return 0.5 * (act(a1) + act(a2))


def make_runs(scenario, n_people, rng):
    """Simulate n_people from one cohort scenario across all drugs and environments."""
    cohort = pd.read_csv(DATA_DIR / f"cohort_cyp2c19_{scenario}_cov.csv")
    people = cohort.sample(n=n_people, random_state=int(rng.integers(1_000_000)))
    params = {d: load_params(d) for d in DRUGS}
    ranges = {d: load_ranges(d) for d in DRUGS}
    rows = []
    for person in people.itertuples():
        for drug in DRUGS:
            p = dict(params[drug])
            lo_hi = ranges[drug]
            p["fm"] = rng.uniform(*lo_hi["fm"])          # sampled per run (uncertainty)
            p["a17"] = rng.uniform(*lo_hi["a17"])
            activity = activity_of(person.allele_1, person.allele_2, p["a17"])
            for env_name in ENVS:
                env = load_environment(env_name, rng=rng)
                res = simulate(p, activity=activity, env=env,
                               weight_kg=person.weight_kg, t_end=48.0)
                row = {
                    "scenario": scenario, "person_id": person.person_id,
                    "phenotype": person.phenotype, "sex": int(person.sex == "male"),
                    "age_years": person.age_years, "weight_kg": person.weight_kg,
                    "drug": drug, "dose_mg": res["dose"], "environment": env_name,
                    "gravity_g": GRAVITY[env_name], "fm": p["fm"], "a17": p["a17"],
                    "activity": activity,
                    "AUC": res["AUC"], "Cmax": res["Cmax"], "tmax": res["tmax"],
                }
                for k in ENV_PARAMS:
                    row["mult_" + k] = env.get(k, 1.0)
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n_people = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    scenarios = sys.argv[2:] or ["pk2021", "riaz2019"]
    rng = np.random.default_rng(SEED)
    parts = []
    for sc in scenarios:
        print(f"Simulating {n_people} people from {sc} ...", flush=True)
        parts.append(make_runs(sc, n_people, rng))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(DATA_DIR / "sim_runs.csv", index=False)
    print("Saved data/sim_runs.csv -", len(out), "runs")
    print(out.groupby(["drug", "environment"])[["AUC", "Cmax", "tmax"]].median().round(3))