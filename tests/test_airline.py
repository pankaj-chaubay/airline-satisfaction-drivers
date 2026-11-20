import pandas as pd
import pytest

from airline_satisfaction import data, drivers, model
from airline_satisfaction.__main__ import main


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "sample.csv"
    data.synthetic_sample(4000).to_csv(p, index=False)
    return data.load(p)


def test_load_accepts_both_label_names(tmp_path):
    raw = data.synthetic_sample(200)
    p = tmp_path / "a.csv"
    raw.rename(columns={"satisfaction": "customer_satisfaction"}).to_csv(p, index=False)
    assert set(data.load(p)[data.TARGET].unique()) <= {0, 1}


def test_load_rejects_missing_columns(tmp_path):
    p = tmp_path / "b.csv"
    data.synthetic_sample(50).drop(columns=["Cleanliness"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Cleanliness"):
        data.load(p)


def test_split_is_stratified_and_disjoint(df):
    tr, va, te = data.split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    for part in (tr, va, te):
        assert abs(part[data.TARGET].mean() - df[data.TARGET].mean()) < 0.02


def test_models_beat_chance_on_sample(df):
    tr, va, te = data.split(df)
    best, fitted, table = model.select(tr, va)
    assert best in fitted
    assert table["roc_auc"].min() > 0.7


def test_importance_finds_the_planted_driver(df):
    tr, va, te = data.split(df)
    best, fitted, _ = model.select(tr, va)
    imp = drivers.importance(fitted[best], te, n_repeats=3)
    assert imp.loc[0, "feature"] == "Inflight entertainment"
    assert imp["cumulative_share"].iloc[-1] == pytest.approx(1.0, abs=1e-3)


def test_rating_gaps_ignore_zero(df):
    g = drivers.rating_gaps(df)
    assert set(g["service"]) == set(data.RATINGS)
    assert g["gap"].is_monotonic_decreasing


def test_scores():
    s = model.scores([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.6])
    assert s["accuracy"] == 0.5 and s["tp"] == 1 and s["fp"] == 1


def test_cli_runs_on_sample(tmp_path):
    main(["--sample", "--out", str(tmp_path)])
    text = (tmp_path / "results.md").read_text()
    assert "Test set" in text and (tmp_path / "figures" / "drivers.png").exists()
