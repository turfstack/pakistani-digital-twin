"""Pakistani Pharmacological Digital Twin: Streamlit app.

Run from the project folder with:  streamlit run app.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from add_covariates import add_covariates                       # noqa: E402
from environments import GRAVITY, load_environment              # noqa: E402
from make_sim_runs import ENV_PARAMS, load_ranges               # noqa: E402
from pbpk import load_params, simulate                          # noqa: E402
from train_twin import FEATURES, Twin, make_mlp                 # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "outputs"
SEED = 42
N_TWIN = 3000        # people per population for the AI twin
N_CURVE = 100        # people for the PBPK concentration curves
DRUGS = {"Omeprazole": "omeprazole", "Acetaminophen": "acetaminophen"}
ENVS = {"Earth": "earth", "Space (microgravity)": "space", "Mars (0.38 g)": "mars"}
POPS = {"Pakistani, Riaz 2019 frequencies": "riaz2019",
        "Pakistani, 2021 study": "pk2021",
        "European reference (approximate)": "reference"}
POP_SHORT = {"riaz2019": "Riaz 2019", "pk2021": "2021 study", "reference": "European ref."}
REFERENCE_FREQ = {"*1": 0.65, "*2": 0.15, "*17": 0.20}   # approximate; verify before publishing
ENV_COLOR = {"earth": "#6fb7ff", "space": "#c8a2ff", "mars": "#ff9a5c"}
ENV_STYLE = {"earth": "--", "space": ":", "mars": "-"}
EVIDENCE = {"earth": "Partly validated against published pharmacokinetic data.",
            "space": "Low to medium evidence: only the absorption-speed change has a source (a bed-rest study); other effects are placeholder ranges.",
            "mars": "Very low evidence: a linear interpolation of the assumed space effects. Uncertainty ranges may be too narrow."}

st.set_page_config(page_title="Pakistani Digital Twin", layout="wide")
st.markdown("""<style>
.card{background:#121a30;border:1px solid #243055;border-radius:10px;padding:14px 18px;margin-bottom:12px}
.card h4{margin:0 0 6px 0;font-size:17px}
.card p{margin:0;font-size:15px;line-height:1.45;color:#cdd6ea}
.pill{display:inline-block;border:1px solid #36508a;border-radius:999px;padding:5px 12px;margin:2px;background:#121a30;font-size:14px}
.bar{display:flex;align-items:center;gap:10px;font-size:14px;min-height:26px}
.bar b{width:30px}.bar i{display:block;height:10px;border-radius:5px;background:linear-gradient(90deg,#3a8fd0,#5cc8ff)}
.bar span{color:#9aa7c4}
button[kind="primary"],[data-testid="stBaseButton-primary"]{color:#06121f !important;font-weight:700}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource
def get_twin():
    """Rebuild the trained AI twin from outputs/twin_ensemble.pt."""
    ck = torch.load(OUT / "twin_ensemble.pt", weights_only=False)
    twin = Twin()
    twin.xm, twin.xs, twin.ym, twin.ys = ck["xm"], ck["xs"], ck["ym"], ck["ys"]
    twin.models = []
    for state in ck["states"]:
        m = make_mlp(len(FEATURES))
        m.load_state_dict(state)
        twin.models.append(m.eval())
    return twin


@st.cache_data(show_spinner=False)
def get_cohort(key, n=N_TWIN):
    """A random sample of people from one population."""
    if key == "reference":
        rng = np.random.default_rng(SEED)
        a = rng.choice(list(REFERENCE_FREQ), size=(n, 2), p=list(REFERENCE_FREQ.values()))
        df = pd.DataFrame({"person_id": np.arange(n), "allele_1": a[:, 0], "allele_2": a[:, 1]})
        return add_covariates(df).reset_index(drop=True)
    path = DATA / f"cohort_cyp2c19_{key}_cov.csv"
    if not path.exists():                                  # deployed copy: small committed sample
        path = DATA / "deploy" / f"cohort_{key}_sample.csv"
    if not path.exists():
        st.error(f"Missing cohort file for {key}. Run:  python src\\add_covariates.py {key}")
        st.stop()
    df = pd.read_csv(path)
    return df.sample(n=min(n, len(df)), random_state=SEED).reset_index(drop=True)


# ----------------------------------------------------------- feature building
def env_multipliers(env, n, rng):
    """Sampled environment multipliers (Earth = 1; Mars = partial space effect)."""
    out = {k: np.ones(n) for k in ENV_PARAMS}
    if env == "earth":
        return out
    strength = 1.0 - GRAVITY[env]
    tab = pd.read_csv(DATA / "environments.csv")
    for r in tab[tab["environment"] == "space"].itertuples():
        out[r.parameter] = 1.0 + (rng.uniform(r.low, r.high, n) - 1.0) * strength
    return out


def build_features(cohort, drug, env, rng, fm=None, a17=None):
    """Inputs for the AI twin (same columns it was trained on)."""
    n, p, rg = len(cohort), load_params(drug), load_ranges(drug)
    fm = rng.uniform(*rg["fm"], n) if fm is None else fm
    a17 = rng.uniform(*rg["a17"], n) if a17 is None else a17
    base = {"*1": 1.0, "*2": 0.0, "*3": 0.0}
    a1 = np.where(cohort["allele_1"] == "*17", a17, cohort["allele_1"].map(base).to_numpy(dtype=float))
    a2 = np.where(cohort["allele_2"] == "*17", a17, cohort["allele_2"].map(base).to_numpy(dtype=float))
    mult = env_multipliers(env, n, rng)
    return pd.DataFrame({
        "drug_code": float(drug == "acetaminophen"), "dose_mg": p["dose_mg"],
        "gravity_g": GRAVITY[env], "sex": (cohort["sex"] == "male").astype(float).to_numpy(),
        "age_years": cohort["age_years"].to_numpy(), "weight_kg": cohort["weight_kg"].to_numpy(),
        "activity": 0.5 * (a1 + a2), "fm": fm, "a17": a17,
        **{"mult_" + k: v for k, v in mult.items()}})


def twin_predict(df):
    mean, std = get_twin().predict(df[FEATURES])
    return np.exp(mean[:, 0]), np.exp(mean[:, 1]), mean[:, 2]      # AUC, Cmax, tmax


def pbpk_curves(df, drug, n):
    """PBPK concentration curves for the first n people in df."""
    p0, curves, t = load_params(drug), [], None
    for r in df.head(n).itertuples(index=False):
        p = dict(p0, fm=r.fm, a17=r.a17)
        env = {k: getattr(r, "mult_" + k) for k in ENV_PARAMS}
        res = simulate(p, activity=r.activity, env=env, weight_kg=r.weight_kg, t_end=12.0)
        t, _ = res["t"], curves.append(res["conc"])
    return t, np.array(curves)


# ---------------------------------------------------------------- simulation
@st.cache_data(show_spinner="Running simulation ...")
def population_results(drug, env, pop):
    rng = np.random.default_rng(SEED)
    cohorts = {k: get_cohort(k) for k in POP_SHORT}
    ref_auc, _, _ = twin_predict(build_features(cohorts["reference"], drug, "earth", rng))
    thr = np.quantile(ref_auc, 0.90)

    shares, sel = {}, None
    for k, c in cohorts.items():
        df = build_features(c, drug, env, rng)
        auc, cmax, tmax = twin_predict(df)
        shares[k] = float(np.mean((ref_auc if (k == 'reference' and env == 'earth') else auc) > thr) * 100)
        if k == pop:
            sel = (df, auc, cmax, tmax)
    df, auc, cmax, tmax = sel
    q = lambda a: np.percentile(a, [5, 50, 95])
    # PBPK curves for the selected population: chosen environment and Earth (same people)
    t, c_env = pbpk_curves(df, drug, N_CURVE)
    earth_df = build_features(cohorts[pop].head(N_CURVE), drug, "earth", rng,
                              fm=df["fm"].to_numpy()[:N_CURVE], a17=df["a17"].to_numpy()[:N_CURVE])
    _, c_earth = pbpk_curves(earth_df, drug, N_CURVE)
    pct = lambda c: np.percentile(c, [5, 50, 95], axis=0)
    return {"auc": q(auc), "cmax": q(cmax), "tmax": q(tmax), "shares": shares, "thr": float(thr),
            "t": t, "env_curve": pct(c_env), "earth_curve": pct(c_earth)}


@st.cache_data(show_spinner=False)
def person_curves(phenotype, drug):
    """One reference adult (70 kg) of a given metabolizer type in the three environments."""
    p = load_params(drug)
    a17 = load_params("omeprazole")["a17"]
    activity = {"PM": 0.0, "IM": 0.5, "NM": 1.0, "RM": (1 + a17) / 2, "UM": a17}[phenotype]
    out = {}
    for env in ENV_COLOR:
        res = simulate(p, activity=activity, env=load_environment(env), t_end=12.0)
        out[env] = (res["t"], res["conc"])
    return out


def style(ax, ylabel="Concentration (mg/L)"):
    ax.set_facecolor("#121a30")
    ax.figure.set_facecolor("#121a30")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#5b6a90")
    ax.tick_params(colors="#9aa7c4")
    ax.set_xlabel("Hours after dose", color="#9aa7c4")
    ax.set_ylabel(ylabel, color="#9aa7c4")


COMPARTMENTS_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 270" width="540" height="270" font-family="sans-serif">
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0 0L10 5L0 10Z" fill="#5cc8ff"/></marker></defs>
<g fill="#0f1630" stroke="#36508a" stroke-width="1.5"><rect x="70" y="60" width="100" height="60" rx="8"/>
<rect x="240" y="60" width="100" height="60" rx="8"/><rect x="410" y="60" width="100" height="60" rx="8"/>
<rect x="410" y="190" width="100" height="60" rx="8"/></g>
<g fill="none" stroke="#5cc8ff" stroke-width="1.8" marker-end="url(#ar)"><path d="M8 90H66"/><path d="M174 90H236"/>
<path d="M344 90H406" marker-start="url(#ar)"/><path d="M460 124V186" marker-start="url(#ar)"/><path d="M290 124V186"/></g>
<g text-anchor="middle" fill="#9aa7c4" font-size="12.5"><g fill="#fff" font-size="16" font-weight="600">
<text x="120" y="88">Gut</text><text x="290" y="88">Liver</text><text x="460" y="88">Blood</text><text x="460" y="218">Tissue</text></g>
<g font-size="12"><text x="120" y="104">absorption site</text><text x="290" y="104">metabolism</text>
<text x="460" y="104">central</text><text x="460" y="234">peripheral</text></g>
<text x="37" y="78">Oral dose</text><text x="205" y="78">absorbed</text><text x="375" y="78">blood flow</text>
<text x="290" y="208">Metabolized</text><text x="290" y="222">(CYP2C19 for omeprazole)</text>
<text x="450" y="160" text-anchor="end">exchange</text></g></svg>"""


# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    drug_label = st.radio("Drug", list(DRUGS))
    env_label = st.radio("Environment", list(ENVS), index=2)
    pop_label = st.selectbox("Population", list(POPS))
    run = st.button("Run simulation", type="primary")

drug, env, pop = DRUGS[drug_label], ENVS[env_label], POPS[pop_label]
current = (drug, env, pop)
if "active" not in st.session_state or run:
    st.session_state["active"] = current
active = st.session_state["active"]

st.title("Pakistani Pharmacological Digital Twin")
st.caption("Research prototype. Not for dosing decisions.")
tab1, tab2, tab3 = st.tabs(["Model view", "Charts and results", "About this project"])

# ----------------------------------------------------------------- tab 1
with tab1:
    if env != "earth":
        st.warning(f"**{env_label.split(' (')[0]} is assumption-based.** {EVIDENCE[env]}")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Model compartments")
        st.image(COMPARTMENTS_SVG)
        st.caption("The four compartments of the PBPK model. The diagram is schematic.")
    with right:
        st.subheader("Metabolizer type (CYP2C19)")
        pheno = st.radio("Metabolizer type", ["PM", "IM", "NM", "RM", "UM"], index=2,
                         horizontal=True, label_visibility="collapsed")
        a17 = load_params("omeprazole")["a17"]
        acts = {"PM": 0.0, "IM": 0.5, "NM": 1.0, "RM": (1 + a17) / 2, "UM": a17}
        st.markdown("".join(
            f'<div class="bar" style="opacity:{1 if k == pheno else 0.35}"><b>{k}</b>'
            f'<i style="width:{int(v / a17 * 150)}px"></i><span>{v:.1f}</span></div>'
            for k, v in acts.items()), unsafe_allow_html=True)
        st.caption("Relative enzyme activity used by the model (the *17 value is uncertain). "
                   "For acetaminophen, genotype has no effect.")
    st.subheader(f"Drug level over 12 hours: {pheno}, 70 kg adult, {drug_label.lower()}")
    curves = person_curves(pheno, drug)
    ymax = 1.08 * max(c.max() for ph in acts for (_, c) in person_curves(ph, drug).values())
    area = lambda ph: np.trapezoid(person_curves(ph, drug)["earth"][1], person_curves(ph, drug)["earth"][0])
    if drug == "acetaminophen":
        st.info("Genotype has no effect on acetaminophen in this model, so every metabolizer type gives the "
                "same curves. Pick omeprazole to see the difference.")
    else:
        st.caption(f"Exposure over 12 h on Earth compared with a normal metabolizer (NM): "
                   f"x{area(pheno) / area('NM'):.1f}. The y-axis is fixed so the types can be compared.")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for e, (t, c) in curves.items():
        ax.plot(t, c, ENV_STYLE[e], color=ENV_COLOR[e], lw=2.8 if e == env else 1.8,
                label=[k for k, v in ENVS.items() if v == e][0])
    style(ax)
    ax.set_ylim(0, ymax)
    ax.legend(facecolor="#121a30", edgecolor="#243055", labelcolor="#e6ebf5")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Single-person PBPK simulation with central parameter values.")

# ----------------------------------------------------------------- tab 2
with tab2:
    if active != current:
        st.info("Settings changed. Press **Run simulation** to update these results.")
    a_drug, a_env, a_pop = active
    res = population_results(a_drug, a_env, a_pop)
    a_env_label = [k for k, v in ENVS.items() if v == a_env][0]
    a_drug_label = [k for k, v in DRUGS.items() if v == a_drug][0]
    st.markdown(f"**{a_drug_label}, {a_env_label}, {[k for k, v in POPS.items() if v == a_pop][0]}**")
    if a_env != "earth":
        st.warning(f"**{a_env_label.split(' (')[0]} is assumption-based.** {EVIDENCE[a_env]}")
    k1, k2, k3, k4 = st.columns(4)
    for col, label, key, fmt in [(k1, "Median AUC (mg·h/L)", "auc", "{:.3g}"),
                                 (k2, "Median Cmax (mg/L)", "cmax", "{:.3g}"),
                                 (k3, "Median tmax (h)", "tmax", "{:.2f}")]:
        lo, mid, hi = res[key]
        col.metric(label, fmt.format(mid))
        col.caption(f"5-95% range: {fmt.format(lo)} to {fmt.format(hi)}")
    k4.metric("Above reference Earth 90th percentile", f"{res['shares'][a_pop]:.0f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drug level over time")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        lo, mid, hi = res["env_curve"]
        ax.fill_between(res["t"], lo, hi, color=ENV_COLOR[a_env], alpha=0.25, lw=0)
        ax.plot(res["t"], mid, color=ENV_COLOR[a_env], lw=2.5, label=f"{a_env_label.split(' (')[0]} median")
        if a_env != "earth":
            ax.plot(res["t"], res["earth_curve"][1], "--", color=ENV_COLOR["earth"], lw=1.8, label="Earth median")
        style(ax)
        ax.legend(facecolor="#121a30", edgecolor="#243055", labelcolor="#e6ebf5")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"PBPK simulation of {N_CURVE} people from the selected population. Band: 90% range.")
    with c2:
        st.subheader("Share above reference threshold")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        names = [POP_SHORT[k] for k in POP_SHORT]
        ax.bar(names, [res["shares"][k] for k in POP_SHORT], color=["#5cc8ff", "#8b9dff", "#6b7899"])
        ax.axhline(10, color="#e6ebf5", ls="--", lw=1)
        style(ax, "% of people")
        ax.set_xlabel("")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Dashed line: 10%, the European reference on Earth by definition. "
                   f"Threshold: AUC {res['thr']:.3g} mg·h/L (90th percentile, European reference, Earth).")

    with st.expander("What drives the AI twin's predictions (SHAP)"):
        img = OUT / "shap_importance.png"
        if img.exists():
            st.image(str(img))
            st.caption("Mean absolute SHAP value per input, by drug and output. SHAP explains the AI, not reality.")
        else:
            st.write("Run  python src\\insights.py  to create this figure.")
    st.caption("Synthetic cohorts resembling published CYP2C19 allele frequencies. Predictions come from the AI "
               "twin (a fast copy of the PBPK model); curves come from the PBPK model itself.")

# ----------------------------------------------------------------- tab 3
with tab3:
    st.subheader("About this project")
    st.markdown("".join(f'<span class="pill">{x}</span> →' if i < 4 else f'<span class="pill">{x}</span>'
                        for i, x in enumerate(["Published allele frequencies", "Synthetic cohort",
                                               "PBPK model × environment", "AI twin", "Insights and app"])),
                unsafe_allow_html=True)
    about = [
        ("1. The PBPK model", "A physiologically based pharmacokinetic (PBPK) model follows a drug through the body as connected compartments: gut, liver, blood and peripheral tissue. Equations set how fast the drug is absorbed, how much the liver removes on each pass, and how it spreads into tissue. Because the compartments use real physiological quantities (blood flow, volumes, enzyme capacity), an environmental change can be written as a change in one specific quantity. One result follows from the equations: for a liver-cleared drug, oral exposure (AUC) depends on how much is absorbed and on enzyme capacity, not on blood flow, volumes or absorption speed. Those change the peak and its timing instead."),
        ("2. The population simulations, and why", "Individual-level genotype data for Pakistani populations is too limited for this question, so we built a synthetic cohort of one million virtual people. Their CYP2C19 variants are drawn from published Pakistani allele frequencies, which gives each person a metabolizer type (poor to ultrarapid) alongside age, weight and sex. Two published studies disagree, so each is kept as its own scenario, and every simulated cohort re-draws the frequencies to carry that uncertainty forward. Why: one average person hides the people most at risk. Simulating a population shows how many would end up with unusually high or low drug levels."),
        ("3. The AI twins, and what they do", "Running the PBPK equations for every virtual person, drug and environment is slow. We trained a small ensemble of neural networks on the PBPK results so that it predicts peak level, timing and total exposure almost instantly, with an uncertainty range. On new virtual people it matches the PBPK to about 1%. It is a fast copy of the model, not a new source of biology: it can be no more correct than the PBPK it learned from, and outside the conditions it was trained on its uncertainty range is too narrow, which is why Mars results carry a warning. Its explanations (SHAP) show which factors drive each result."),
        ("4. The point of this project", "To show, end to end, how genetics, physiology and environment can be combined to ask who might respond differently to a drug, and when, using Pakistani population data instead of assuming other populations apply. It is a transparent, extensible framework: more drugs, genes and ethnic groups can be added as new inputs. It generates hypotheses for study. It is not a clinical or flight tool, space and Mars effects are literature-based assumptions without validation, and nothing here should inform dosing."),
    ]
    for row in (about[:2], about[2:]):
        for col, (title, text) in zip(st.columns(2), row):
            col.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)