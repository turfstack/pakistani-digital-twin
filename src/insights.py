"""Win 8: (1) SHAP explanations of the AI twin, (2) Pakistani vs reference exposure comparison."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environments import GRAVITY, load_environment  # noqa: E402
from make_sim_runs import ENV_PARAMS, load_ranges  # noqa: E402
from pbpk import load_params  # noqa: E402
from train_twin import FEATURES, TARGETS, Twin, make_mlp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SEED = 42
DRUGS = ["omeprazole", "acetaminophen"]
ENVS = ["earth", "space", "mars"]

# European reference allele frequencies from the CPIC clopidogrel guideline, as tabulated in Expert Rev Clin Pharmacol 2017 Table 1 (PMC5771818); *1 is the remainder so the four add up to exactly 1
REFERENCE_FREQ = {"*1": 0.6358, "*2": 0.15, "*3": 0.0042, "*17": 0.21}


def load_twin():
    """Rebuild the saved ensemble (weights + scaling) from outputs/twin_ensemble.pt."""
    ck = torch.load(OUT / "twin_ensemble.pt", weights_only=False)
    twin = Twin()
    twin.xm, twin.xs, twin.ym, twin.ys = ck["xm"], ck["xs"], ck["ym"], ck["ys"]
    twin.models = []
    for state in ck["states"]:
        m = make_mlp(len(FEATURES))
        m.load_state_dict(state)
        twin.models.append(m.eval())
    return twin


# ---------------------------------------------------------------- SHAP
def run_shap(twin, n_background=100, n_explain=150):
    """Mean |SHAP| per feature for each drug and target (original target units)."""
    test = pd.read_csv(OUT / "test_predictions.csv")
    rows = []
    for drug in DRUGS:
        d = test[test["drug"] == drug]
        bg = d.sample(n=min(n_background, len(d)), random_state=SEED)[FEATURES]
        ex = d.sample(n=min(n_explain, len(d)), random_state=SEED + 1)[FEATURES]
        predict = lambda X: twin.predict(pd.DataFrame(X, columns=FEATURES))[0]
        sv = shap.KernelExplainer(predict, bg.values).shap_values(ex.values, nsamples=200)
        sv = np.array(sv)
        if sv.shape[0] == len(ex):            # (rows, features, targets) in some shap versions
            sv = np.moveaxis(sv, -1, 0)       # -> (targets, rows, features)
        for j, t in enumerate(TARGETS):
            for f, v in zip(FEATURES, np.abs(sv[j]).mean(axis=0)):
                rows.append({"drug": drug, "target": t, "feature": f, "mean_abs_shap": v})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "shap_importance.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, drug in enumerate(DRUGS):
        for j, t in enumerate(TARGETS):
            s = table[(table.drug == drug) & (table.target == t)].sort_values("mean_abs_shap").tail(8)
            axes[i, j].barh(s["feature"], s["mean_abs_shap"])
            axes[i, j].set_title(f"{drug}: {t}")
    plt.tight_layout()
    plt.savefig(OUT / "shap_importance.png", dpi=130)
    plt.close()
    return table


# ---------------------------------------------------- population comparison
def reference_cohort(n, rng):
    """Synthetic reference cohort from approximate European allele frequencies."""
    alleles, probs = list(REFERENCE_FREQ), list(REFERENCE_FREQ.values())
    a = rng.choice(alleles, size=(n, 2), p=probs)
    df = pd.DataFrame({"person_id": np.arange(n), "allele_1": a[:, 0], "allele_2": a[:, 1]})
    from add_covariates import add_covariates
    return add_covariates(df)


def twin_rows(cohort, scenario, rng):
    """Build twin inputs (people x drugs x environments) with sampled fm, a17, multipliers."""
    rows = []
    ranges = {d: load_ranges(d) for d in DRUGS}
    params = {d: load_params(d) for d in DRUGS}
    act = {"*1": 1.0, "*2": 0.0, "*3": 0.0}
    for drug in DRUGS:
        n = len(cohort)
        fm = rng.uniform(*ranges[drug]["fm"], n)
        a17 = rng.uniform(*ranges[drug]["a17"], n)
        a1 = cohort["allele_1"].map(lambda x: act.get(x, np.nan)).to_numpy()
        a2 = cohort["allele_2"].map(lambda x: act.get(x, np.nan)).to_numpy()
        a1 = np.where(cohort["allele_1"] == "*17", a17, a1)
        a2 = np.where(cohort["allele_2"] == "*17", a17, a2)
        for env in ENVS:
            mult = {k: np.ones(n) for k in ENV_PARAMS}
            if env != "earth":
                for k in ENV_PARAMS:
                    mult[k] = np.array([load_environment(env, rng=rng).get(k, 1.0) for _ in range(n)])
            df = pd.DataFrame({
                "population": scenario, "drug": drug, "environment": env,
                "drug_code": float(drug == "acetaminophen"), "dose_mg": params[drug]["dose_mg"],
                "gravity_g": GRAVITY[env], "sex": (cohort["sex"] == "male").astype(float).to_numpy(),
                "age_years": cohort["age_years"].to_numpy(), "weight_kg": cohort["weight_kg"].to_numpy(),
                "activity": 0.5 * (a1 + a2), "fm": fm, "a17": a17,
                **{"mult_" + k: v for k, v in mult.items()}})
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def compare_populations(twin, n=2000):
    rng = np.random.default_rng(SEED)
    pops = {"pakistan_riaz2019": pd.read_csv(ROOT / "data" / "cohort_cyp2c19_riaz2019_cov.csv").sample(n, random_state=1),
            "pakistan_pk2021": pd.read_csv(ROOT / "data" / "cohort_cyp2c19_pk2021_cov.csv").sample(n, random_state=2),
            "reference_european": reference_cohort(n, rng)}
    frames = []
    for name, cohort in pops.items():
        d = twin_rows(cohort, name, rng)
        mean, _ = twin.predict(d)
        d["AUC"], d["Cmax"], d["tmax"] = np.exp(mean[:, 0]), np.exp(mean[:, 1]), mean[:, 2]
        frames.append(d)
    allp = pd.concat(frames, ignore_index=True)

    # Threshold: 90th percentile of AUC in the reference population on Earth, per drug.
    ref_earth = allp[(allp.population == "reference_european") & (allp.environment == "earth")]
    thr = ref_earth.groupby("drug")["AUC"].quantile(0.90)
    allp["above"] = allp["AUC"] > allp["drug"].map(thr)
    table = (allp.groupby(["drug", "environment", "population"])["above"].mean() * 100).round(1)
    table = table.unstack("population")
    table.to_csv(OUT / "population_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, drug in zip(axes, DRUGS):
        table.loc[drug].reindex(ENVS).plot.bar(ax=ax, rot=0)
        ax.set_title(f"{drug}: % above reference Earth 90th-percentile AUC")
        ax.set_ylabel("% of people")
    plt.tight_layout()
    plt.savefig(OUT / "population_comparison.png", dpi=130)
    plt.close()
    return table, thr


if __name__ == "__main__":
    twin = load_twin()
    print("Computing SHAP (a few minutes) ...", flush=True)
    shap_table = run_shap(twin)
    for drug in DRUGS:
        for t in TARGETS:
            s = shap_table[(shap_table.drug == drug) & (shap_table.target == t)]
            top = s.sort_values("mean_abs_shap", ascending=False).head(4)
            print(f"{drug:14s} {t:8s} top drivers:",
                  ", ".join(f"{f} ({v:.3f})" for f, v in zip(top.feature, top.mean_abs_shap)))
    print("\nComparing populations ...", flush=True)
    table, thr = compare_populations(twin)
    print("AUC threshold (reference Earth 90th percentile):", thr.round(3).to_dict())
    print(table.to_string())
    print("\nSaved outputs/shap_importance.csv/.png and population_comparison.csv/.png")