# Pakistani Pharmacological Digital Twin

An interactive computational pharmacology prototype simulating **omeprazole** (CYP2C19-dependent) and **acetaminophen** (CYP2C19-independent control) pharmacokinetics across Pakistani **CYP2C19** metabolizer phenotypes and gravitational environments (**Earth**, **Space microgravity**, and **Mars 0.38 g**).

**Live App:** [pakistani-digital-twin.streamlit.app](https://pakistani-digital-twin.streamlit.app/)

## Overview
1. **Multi-Compartment PBPK Engine (`src/pbpk.py`):** Solves a 4-compartment ODE system (**gut $\rightarrow$ liver $\rightarrow$ central blood $\leftrightarrow$ peripheral tissue**, plus mass-balance and cumulative AUC states via `scipy.integrate.solve_ivp` `LSODA`) parameterized for a 70 kg reference adult with allometric scaling ($W^{0.75}$ for flows/clearance, $W^{1.0}$ for volumes) and Earth ($1.0\,g$), Space microgravity, and Mars ($0.38\,g$) physiological multipliers.
2. **Population Simulations (`src/make_cohort.py`):** Generates **1,000,000 virtual individuals per cohort** (`10 replicates × 100,000`) drawing CYP2C19 alleles (`*1`, `*2`, `*3`, `*17`) with Dirichlet inter-study uncertainty (`effective_n = 200` chromosomes) from published Pakistani allele frequency datasets (`riaz2019` and `pk2021`, modeled separately rather than pooled) alongside a European reference cohort, paired with age, weight, and sex covariates (`data/deploy/` stores 3,000-person subsamples for web deployment).
3. **AI Surrogate Ensemble (`src/train_twin.py`):** A 5-model PyTorch MLP deep ensemble (`twin_ensemble.pt`) trained with person-level 70/15/15 splits (`pkey`-grouped to prevent leakage) on PBPK simulation runs to predict $\ln(\text{AUC})$, $\ln(C_{\max})$, and $T_{\max}$ alongside epistemic uncertainty intervals and SHAP feature attribution (`src/insights.py`).
4. **Interactive Streamlit App (`app.py`):** Compare single-person 12-hour concentration-time curves and 3,000-person population exposure distributions (against the 90th-percentile European Earth reference threshold) across **Omeprazole** and **Acetaminophen**, CYP2C19 phenotypes (`PM`, `IM`, `NM`, `RM`, `UM`), and environments (**Earth**, **Space**, **Mars**).

## Surrogate Model Benchmark (`outputs/twin_metrics.csv`)
Evaluated on strictly held-out virtual individuals (`pkey`-grouped split) against linear regression and gradient-boosted tree (`HistGradientBoostingRegressor`) baselines:

| Evaluation Regime | Target | RMSE | $R^2$ | Within $1.25\times$ | 90% Interval Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Held-Out People (All Environments)** | $\ln(\text{AUC})$ | **0.0113** (~1.1% error) | 1.0000 | 100.0% | 96.6% |
| **Held-Out People (All Environments)** | $\ln(C_{\max})$ | **0.0087** (~0.9% error) | 1.0000 | 100.0% | 96.9% |
| **Held-Out People (All Environments)** | $T_{\max}$ (h) | **0.0034** | 0.9998 | 100.0% | 83.6% |
| **Cross-Cohort (`riaz2019` $\rightarrow$ `pk2021`)** | $\ln(\text{AUC})$ | **0.0206** | 0.9999 | 99.9% | 96.6% |
| **Zero-Shot Extrapolation (Mars $0.38\,g$)** | $\ln(\text{AUC})$ | **0.1439** | 0.9966 | 97.7% | 18.0% |

*Key finding:* While point predictions remain within $1.25\times$ bioequivalence bounds for $97.7\%$ of unseen Mars ($0.38\,g$) simulations, empirical 90% ensemble coverage drops from $96.6\%$ to $18.0\%$, illustrating deep-ensemble overconfidence under out-of-distribution gravity extrapolation.

## Repository Structure
- `app.py` — Streamlit web application (`Model view`, `Charts and results`, and `About this project` tabs)
- `src/` — Cohort generation (`make_cohort.py`), covariate modeling (`add_covariates.py`), PBPK ODE engine (`pbpk.py`, `make_sim_runs.py`), environmental multipliers (`environments.py`), neural surrogate training (`train_twin.py`), SHAP insights (`insights.py`), and deployment sampling (`make_deploy_samples.py`)
- `data/` — Allele frequencies (`allele_frequencies.csv`), drug parameters (`drug_parameters.csv`), environmental multipliers (`environments.csv`), simulation runs (`sim_runs.csv`), and deployment sample cohorts (`deploy/`)
- `outputs/` — Trained PyTorch ensemble (`twin_ensemble.pt`), benchmark metrics (`twin_metrics.csv`), SHAP importance values/plots, and phenotype comparison figures
- `test/` — Unit tests for the PBPK engine (`test_pbpk.py`)

## Disclaimer
This project is a research and educational prototype built on semi-synthetic cohorts derived from published population genetics literature. It is not a clinical dosing or flight tool: Space parameters rely on limited analog data (such as Gandia et al., 2003 $-6^\circ$ head-down bed rest for absorption rate) plus bounded placeholder ranges, and Mars ($0.38\,g$) parameters are unvalidated linear interpolations.

## Development & AI Assistance Disclosure
This prototype was designed, parameterized, and validated by the author, combining domain expertise in clinical pharmacology/pharmacokinetics with applied AI graduate coursework. Large language models (Claude and Gemini) were used as coding and documentation assistants for structuring Python modules (`src/`), Streamlit UI scaffolding (`app.py`), and unit tests, with all scientific assumptions, allele frequency citations, and model outputs reviewed by the author.
