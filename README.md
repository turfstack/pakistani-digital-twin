# Pakistani Pharmacological Digital Twin

An interactive computational pharmacology prototype simulating **omeprazole** and **acetaminophen** pharmacokinetics across Pakistani **CYP2C19** metabolizer phenotypes and gravitational environments (**Earth**, **Space microgravity**, and **Mars 0.38 g**).

**Live App:** [pakistani-digital-twin.streamlit.app](https://pakistani-digital-twin.streamlit.app/)

## Overview
1. **Multi-Compartment PBPK Model:** Tracks drug absorption, hepatic clearance, blood flow, organ volumes, and tissue distribution across gut, liver, blood, and peripheral compartments.
2. **Population Simulations (1,000,000 Virtual Cohort):** Draws CYP2C19 variants (`*1`, `*2`, `*3`, `*17`) from published Pakistani allele frequency studies (including Riaz 2019 and PK2021 cohorts) alongside European reference frequencies (PharmGKB), propagating inter-study and sampling uncertainty across age, weight, and sex distributions.
3. **AI Surrogate Ensemble:** A PyTorch neural network ensemble trained on PBPK simulation outputs to predict peak concentration ($C_{\max}$), timing ($T_{\max}$), and total exposure ($\text{AUC}$) within ~1% of the mechanistic model on unseen virtual individuals, paired with SHAP feature importance analysis.
4. **Interactive Streamlit App:** Compare single-person and population concentration-time profiles across drugs (**Omeprazole**, **Acetaminophen**), CYP2C19 phenotypes, and environments (**Earth**, **Space**, **Mars**).

## Repository Structure
- `app.py` — Streamlit web application
- `src/` — Cohort generation, PBPK simulation engine, covariate modeling, and neural surrogate training scripts
- `data/` — Allele frequencies, drug parameters, environmental scenarios, and deployment sample cohorts
- `outputs/` — Trained PyTorch ensemble (`twin_ensemble.pt`), evaluation metrics, SHAP values, and summary figures
- `test/` — Unit tests for the PBPK engine (`test_pbpk.py`)

## Disclaimer
This project is a research and educational prototype built on semi-synthetic cohorts derived from published population genetics literature. It is not a clinical dosing or flight tool, and out-of-distribution extrapolations (such as Space and Mars physiological parameters) rely on unvalidated literature-based assumptions.

## Development & AI Assistance Disclosure
This prototype was designed, parameterized, and validated by the author, combining domain expertise in clinical pharmacology/pharmacokinetics with applied AI graduate coursework. Large language models (Claude and Gemini) were used as coding and documentation assistants for structuring Python modules (`src/`), Streamlit UI scaffolding (`app.py`), and unit tests, with all scientific assumptions, allele frequency citations, and model outputs reviewed by the author.
