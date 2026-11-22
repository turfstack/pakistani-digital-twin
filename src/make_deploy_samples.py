"""Save small cohort samples (3,000 people each) so the app can run without the 1,000,000-person files."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
out = DATA / "deploy"
out.mkdir(exist_ok=True)
for key in ["riaz2019", "pk2021"]:
    df = pd.read_csv(DATA / f"cohort_cyp2c19_{key}_cov.csv").sample(n=3000, random_state=42)
    df.to_csv(out / f"cohort_{key}_sample.csv", index=False)
    print("Saved", out / f"cohort_{key}_sample.csv", len(df), "people")