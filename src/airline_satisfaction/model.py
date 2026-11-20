"""Three classifiers, chosen on the validation set and tested once on the test set."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, RATINGS, TARGET


def _logistic() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC + RATINGS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])


def _forest(seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC + RATINGS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed))])


def _boosting(seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC + RATINGS),  # missing values handled natively
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ]
    )
    n_num = len(NUMERIC + RATINGS)
    cats = [False] * n_num + [True] * len(CATEGORICAL)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, categorical_features=cats, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def candidates(seed: int = 42) -> dict[str, Pipeline]:
    return {"logistic_regression": _logistic(), "random_forest": _forest(seed), "gradient_boosting": _boosting(seed)}


def scores(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select(train: pd.DataFrame, valid: pd.DataFrame, seed: int = 42) -> tuple[str, dict[str, Pipeline], pd.DataFrame]:
    fitted, rows = {}, []
    for name, pipe in candidates(seed).items():
        pipe.fit(train[FEATURES], train[TARGET])
        fitted[name] = pipe
        rows.append({"model": name, **scores(valid[TARGET], pipe.predict_proba(valid[FEATURES])[:, 1])})
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return table.loc[0, "model"], fitted, table
