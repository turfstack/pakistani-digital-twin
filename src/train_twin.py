"""Train and evaluate the AI twin: a deep ensemble of MLPs emulating the PBPK engine."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pbpk import REF_WEIGHT, clint_total, load_params  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
N_MODELS = 5
FEATURES = ["drug_code", "dose_mg", "gravity_g", "sex", "age_years", "weight_kg",
            "activity", "fm", "a17",
            "mult_ka", "mult_fa", "mult_V_cen", "mult_Qh", "mult_CLint", "mult_fu"]
TARGETS = ["logAUC", "logCmax", "tmax"]


def load_data():
    df = pd.read_csv(ROOT / "data" / "sim_runs.csv")
    df["drug_code"] = (df["drug"] == "acetaminophen").astype(float)
    df["logAUC"] = np.log(df["AUC"])
    df["logCmax"] = np.log(df["Cmax"])
    df["pkey"] = df["scenario"] + "_" + df["person_id"].astype(str)   # unique person id
    return df


def split_by_person(df, seed=SEED):
    """70/15/15 split where all runs of one person stay in the same set."""
    rng = np.random.default_rng(seed)
    keys = rng.permutation(df["pkey"].unique())
    n = len(keys)
    train_k, val_k = keys[: int(0.70 * n)], keys[int(0.70 * n): int(0.85 * n)]
    return (df[df["pkey"].isin(train_k)], df[df["pkey"].isin(val_k)],
            df[~df["pkey"].isin(np.concatenate([train_k, val_k]))])


def make_mlp(n_in):
    return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, len(TARGETS)))


def train_one(Xtr, Ytr, Xva, Yva, seed, max_epochs=400, patience=25, batch=256):
    torch.manual_seed(seed)
    model = make_mlp(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    Xtr, Ytr, Xva, Yva = (torch.tensor(a, dtype=torch.float32) for a in (Xtr, Ytr, Xva, Yva))
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i: i + batch]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva), Yva).item()
        if val < best - 1e-6:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model


class Twin:
    """Ensemble + the scaling it was trained with."""

    def fit(self, train, val, n_models=N_MODELS):
        self.xm, self.xs = np.array(train[FEATURES].mean()), np.array(train[FEATURES].std())
        self.xs[self.xs == 0] = 1.0
        self.ym, self.ys = np.array(train[TARGETS].mean()), np.array(train[TARGETS].std())
        Xtr, Xva = self._x(train), self._x(val)
        Ytr, Yva = (train[TARGETS].values - self.ym) / self.ys, (val[TARGETS].values - self.ym) / self.ys
        self.models = [train_one(Xtr, Ytr, Xva, Yva, seed=SEED + i) for i in range(n_models)]
        return self

    def _x(self, df):
        return (df[FEATURES].values - self.xm) / self.xs

    def predict(self, df):
        X = torch.tensor(self._x(df), dtype=torch.float32)
        with torch.no_grad():
            preds = np.stack([m.eval()(X).numpy() for m in self.models])
        preds = preds * self.ys + self.ym            # back to original target units
        return preds.mean(axis=0), preds.std(axis=0)


def score(name, test, mean, std=None):
    """Metrics per target; ratios are computed on the original (not log) scale."""
    rows = []
    for j, t in enumerate(TARGETS):
        y, yhat = test[t].values, mean[:, j]
        ratio = np.exp(yhat - y) if t.startswith("log") else yhat / y
        row = {"set": name, "target": t,
               "RMSE": float(np.sqrt(np.mean((yhat - y) ** 2))),
               "R2": float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)),
               "within_1.25x": float(np.mean((ratio > 0.8) & (ratio < 1.25))),
               "within_2x": float(np.mean((ratio > 0.5) & (ratio < 2.0)))}
        if std is not None:
            half = 1.645 * std[:, j]
            row["coverage90"] = float(np.mean((y >= yhat - half) & (y <= yhat + half)))
        rows.append(row)
    return rows


def closed_form_logauc(df):
    """Baseline: AUC = fa*Dose/(fu*CLint) from the PBPK identity (needs the physiology)."""
    out = []
    for r in df.itertuples():
        p = load_params(r.drug)
        p["fm"] = r.fm
        size = r.weight_kg / REF_WEIGHT
        fa = min(p["fa"] * r.mult_fa, 1.0)
        clint = clint_total(p, r.activity) * size ** 0.75 * r.mult_CLint
        out.append(np.log(fa * r.dose_mg / (p["fu"] * r.mult_fu * clint)))
    return np.array(out)


def threshold_metric(train, test, mean):
    """Flag people above each drug's 90th-percentile AUC (from training data)."""
    rows = []
    for drug in test["drug"].unique():
        thr = np.log(train.loc[train["drug"] == drug, "AUC"].quantile(0.90))
        m = (test["drug"] == drug).values
        truth, pred = test["logAUC"].values[m] > thr, mean[m, 0] > thr
        tp = np.sum(truth & pred)
        rows.append({"drug": drug, "n_truly_high": int(truth.sum()),
                     "recall": float(tp / max(truth.sum(), 1)),
                     "precision": float(tp / max(pred.sum(), 1))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_data()
    train, val, test = split_by_person(df)
    print(f"People split -> train {train.pkey.nunique()}, val {val.pkey.nunique()}, "
          f"test {test.pkey.nunique()}  (runs {len(train)}/{len(val)}/{len(test)})", flush=True)
    results = []

    # 1) Main ensemble: held-out people, all environments.
    print("Training main ensemble ...", flush=True)
    twin = Twin().fit(train, val)
    mean, std = twin.predict(test)
    results += score("AI twin: held-out people", test, mean, std)

    # Baselines on the same test people.
    Xtr, Xte = (train[FEATURES].values - twin.xm) / twin.xs, (test[FEATURES].values - twin.xm) / twin.xs
    lin = LinearRegression().fit(Xtr, train[TARGETS].values)
    results += score("Baseline: linear regression", test, lin.predict(Xte))
    gb = np.column_stack([HistGradientBoostingRegressor(random_state=SEED).fit(Xtr, train[t]).predict(Xte)
                          for t in TARGETS])
    results += score("Baseline: gradient boosting", test, gb)
    cf = closed_form_logauc(test)
    results += [r for r in score("Baseline: closed-form formula (AUC only)",
                                 test, np.column_stack([cf, cf, cf])) if r["target"] == "logAUC"]

    # 2) Held-out environment: train on Earth + space, test on Mars (unseen 0.38 g).
    print("Training Mars hold-out ensemble ...", flush=True)
    t2 = Twin().fit(train[train.environment != "mars"], val[val.environment != "mars"])
    mars_test = test[test.environment == "mars"]
    m2, s2 = t2.predict(mars_test)
    results += score("AI twin: held-out Mars (0.38 g)", mars_test, m2, s2)

    # 3) Cross-scenario: train on riaz2019, test on pk2021.
    print("Training cross-scenario ensemble ...", flush=True)
    r_tr, r_va, _ = split_by_person(df[df.scenario == "riaz2019"])
    t3 = Twin().fit(r_tr, r_va)
    pk = df[df.scenario == "pk2021"]
    m3, s3 = t3.predict(pk)
    results += score("AI twin: train riaz2019 -> test pk2021", pk, m3, s3)

    table = pd.DataFrame(results).round(4)
    table.to_csv(OUT / "twin_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print("\nHigh-exposure flagging (top 10% AUC), held-out people:")
    print(threshold_metric(train, test, mean).round(3).to_string(index=False))

    # Save the main twin and its test predictions (used later for SHAP and the app).
    torch.save({"features": FEATURES, "targets": TARGETS, "xm": twin.xm, "xs": twin.xs,
                "ym": twin.ym, "ys": twin.ys, "states": [m.state_dict() for m in twin.models]},
               OUT / "twin_ensemble.pt")
    pred = test.copy()
    for j, t in enumerate(TARGETS):
        pred["pred_" + t], pred["sd_" + t] = mean[:, j], std[:, j]
    pred.to_csv(OUT / "test_predictions.csv", index=False)
    print("\nSaved outputs/twin_metrics.csv, twin_ensemble.pt, test_predictions.csv")