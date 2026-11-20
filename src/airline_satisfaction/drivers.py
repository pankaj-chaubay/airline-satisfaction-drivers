"""What drives satisfaction, and the check that explains my original 99% result."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .data import FEATURES, RATINGS, TARGET


def importance(model, test: pd.DataFrame, seed: int = 42, n_repeats: int = 5, max_rows: int = 10000) -> pd.DataFrame:
    """Permutation importance on held-out data: how much ROC AUC drops when a column is shuffled."""
    sample = test.sample(min(len(test), max_rows), random_state=seed)
    r = permutation_importance(model, sample[FEATURES], sample[TARGET], scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    df = pd.DataFrame({"feature": FEATURES, "auc_drop": r.importances_mean, "std": r.importances_std})
    df = df.sort_values("auc_drop", ascending=False).reset_index(drop=True)
    pos = df["auc_drop"].clip(lower=0)
    df["cumulative_share"] = (pos.cumsum() / pos.sum()).round(3)
    df["auc_drop"] = df["auc_drop"].round(4)
    df["std"] = df["std"].round(4)
    return df


def rating_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """For each service: share of low ratings (1-2) among dissatisfied vs satisfied passengers.

    0 is treated as "not rated" and left out.
    """
    rows = []
    for c in RATINGS:
        rated = df[df[c] > 0]
        low = rated[c] <= 2
        d = float(low[rated[TARGET] == 0].mean())
        s = float(low[rated[TARGET] == 1].mean())
        rows.append({"service": c, "low_share_dissatisfied": round(d, 3), "low_share_satisfied": round(s, 3), "gap": round(d - s, 3)})
    return pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)


def row_order_check(df: pd.DataFrame, model, n: int, seed: int = 42) -> pd.DataFrame:
    """Train and test the same model on the first n rows of the file and on a random n rows.

    If the file is not in random order, the first rows can be easier than the rest, and
    accuracy measured on them says little about the whole data set.
    """
    out = []
    for label, part in [(f"first {n} rows", df.head(n)), (f"random {n} rows", df.sample(n, random_state=seed)), ("all rows", df)]:
        tr, te = train_test_split(part, test_size=0.3, random_state=seed)
        m = clone(model).fit(tr[FEATURES], tr[TARGET])
        out.append({"data": label, "rows": len(part), "satisfied_share": round(float(part[TARGET].mean()), 3),
                    "test_accuracy": round(accuracy_score(te[TARGET], m.predict(te[FEATURES])), 4)})
    return pd.DataFrame(out)


def first_rows_profile(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Compare the first n rows with the rest, to show why they are easier."""
    head, rest = df.head(n), df.iloc[n:]
    cols = ["Inflight entertainment", "Seat comfort", "Age", "Flight Distance", "Arrival Delay in Minutes"]
    return pd.DataFrame(
        {
            "feature": cols,
            "first_rows_mean": [round(float(head[c].mean()), 2) for c in cols],
            "other_rows_mean": [round(float(rest[c].mean()), 2) for c in cols],
        }
    )



def satisfaction_by_level(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col)[TARGET].agg(["mean", "size"]).reset_index()
    g.columns = [col, "satisfied_share", "passengers"]
    g["satisfied_share"] = g["satisfied_share"].round(3)
    return g


def zero_rating_share(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Share of 0 ("not rated") answers in the first n rows vs the rest."""
    head, rest = df.head(n), df.iloc[n:]
    cols = [c for c in RATINGS if (df[c] == 0).mean() > 0.01]
    return pd.DataFrame(
        {
            "rating": cols,
            "zero_share_first_rows": [round(float((head[c] == 0).mean()), 3) for c in cols],
            "zero_share_other_rows": [round(float((rest[c] == 0).mean()), 3) for c in cols],
        }
    )
