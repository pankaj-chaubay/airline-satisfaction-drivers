"""Loading, checking and splitting the airline passenger survey."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "satisfied"
CATEGORICAL = ["Customer Type", "Type of Travel", "Class"]
NUMERIC = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
RATINGS = [
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
]
FEATURES = CATEGORICAL + NUMERIC + RATINGS
LABEL_COLUMNS = ("satisfaction", "customer_satisfaction")  # Kaggle name, and the name used in my coursework


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    label = next((c for c in LABEL_COLUMNS if c in df.columns), None)
    if label is None:
        raise ValueError(f"No label column found. Expected one of {LABEL_COLUMNS}.")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from the file: {missing}")
    values = set(df[label].dropna().unique())
    if not values <= {"satisfied", "dissatisfied"}:
        raise ValueError(f"Unexpected label values: {values}")
    out = df[FEATURES].copy()
    out[TARGET] = (df[label] == "satisfied").astype(int)
    for c in CATEGORICAL:
        out[c] = out[c].astype(str)
    return out


def quality_report(df: pd.DataFrame) -> dict:
    """Numbers that go in the report before any modelling."""
    return {
        "rows": int(len(df)),
        "satisfied_share": round(float(df[TARGET].mean()), 3),
        "missing_arrival_delay": int(df["Arrival Delay in Minutes"].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "zero_ratings_share": {c: round(float((df[c] == 0).mean()), 3) for c in RATINGS if (df[c] == 0).any()},
        "delay_correlation": round(float(df["Departure Delay in Minutes"].corr(df["Arrival Delay in Minutes"])), 3),
    }


def split(df: pd.DataFrame, seed: int = 42):
    """Shuffled, stratified 60/20/20 split. The test set is used once, at the end."""
    train, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=seed)
    valid, test = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=seed)
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def synthetic_sample(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Fake data with the same columns as the Kaggle file. Used by the tests and CI only."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Customer Type": rng.choice(["Loyal Customer", "disloyal Customer"], n, p=[0.8, 0.2]),
            "Type of Travel": rng.choice(["Business travel", "Personal Travel"], n, p=[0.7, 0.3]),
            "Class": rng.choice(["Business", "Eco", "Eco Plus"], n, p=[0.48, 0.45, 0.07]),
            "Age": rng.integers(7, 86, n),
            "Flight Distance": rng.integers(50, 6000, n),
            "Departure Delay in Minutes": rng.exponential(15, n).round(),
        }
    )
    df["Arrival Delay in Minutes"] = (df["Departure Delay in Minutes"] + rng.normal(0, 5, n)).clip(0).round()
    for c in RATINGS:
        df[c] = rng.integers(0, 6, n)
    score = (
        0.9 * df["Inflight entertainment"]
        + 0.5 * df["Seat comfort"]
        + 0.4 * df["Online support"]
        + 0.8 * (df["Customer Type"] == "Loyal Customer")
        - 0.01 * df["Arrival Delay in Minutes"]
        + rng.normal(0, 1, n)
    )
    df["satisfaction"] = np.where(score > score.median(), "satisfied", "dissatisfied")
    df.loc[rng.choice(n, 5, replace=False), "Arrival Delay in Minutes"] = np.nan
    return df
