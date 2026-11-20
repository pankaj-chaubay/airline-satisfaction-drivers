"""python -m airline_satisfaction --data data/Invistico_Airline.csv

Use --sample to run on generated fake data (no download needed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import data, drivers, model  # noqa: E402

BLUE, GREY, INK, INK2, SURFACE = "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e", "#fcfcfb"


def md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def driver_chart(imp: pd.DataFrame, path: Path, top: int = 12) -> None:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.barh(d["feature"], d["auc_drop"], color=BLUE, height=0.6)
    ax.set_xlabel("Drop in ROC AUC when the column is shuffled", color=INK2, fontsize=9)
    ax.set_title("What drives passenger satisfaction (test set)", loc="left", fontsize=11, color=INK)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="airline_satisfaction")
    p.add_argument("--data", default="data/Invistico_Airline.csv")
    p.add_argument("--sample", action="store_true", help="use generated fake data instead of the Kaggle file")
    p.add_argument("--out", default="reports")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--original-rows", type=int, default=23616, help="row count of the file used in the original coursework")
    a = p.parse_args(argv)

    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if a.sample:
        tmp = out / "_sample.csv"
        data.synthetic_sample().to_csv(tmp, index=False)
        df = data.load(tmp)
        tmp.unlink()
        a.original_rows = min(a.original_rows, len(df) // 3)
    else:
        df = data.load(Path(a.data))

    quality = data.quality_report(df)
    train, valid, test = data.split(df, a.seed)
    best, fitted, val_table = model.select(train, valid, a.seed)
    test_scores = pd.DataFrame(
        [{"model": n, **model.scores(test[data.TARGET], m.predict_proba(test[data.FEATURES])[:, 1])} for n, m in fitted.items()]
    )
    imp = drivers.importance(fitted[best], test, a.seed)
    gaps = drivers.rating_gaps(df)
    order = drivers.row_order_check(df, model.candidates(a.seed)["random_forest"], a.original_rows, a.seed)
    profile = drivers.first_rows_profile(df, a.original_rows)
    zeros = drivers.zero_rating_share(df, a.original_rows)
    top = imp.loc[imp["feature"].isin(data.RATINGS), "feature"].iloc[0]
    levels = drivers.satisfaction_by_level(df, top)

    val_table.to_csv(out / "validation_scores.csv", index=False)
    test_scores.to_csv(out / "test_scores.csv", index=False)
    imp.to_csv(out / "drivers.csv", index=False)
    gaps.to_csv(out / "rating_gaps.csv", index=False)
    order.to_csv(out / "row_order_check.csv", index=False)
    levels.to_csv(out / "satisfaction_by_top_rating.csv", index=False)
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    driver_chart(imp, out / "figures" / "drivers.png")

    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    report = [
        "# Results",
        "",
        f"Data: {'generated sample (not real)' if a.sample else Path(a.data).name}, {quality['rows']:,} rows, "
        f"{quality['satisfied_share']:.0%} satisfied. Split 60/20/20, stratified, seed {a.seed}.",
        "",
        "## Data checks",
        "",
        f"- Missing values: {quality['missing_arrival_delay']} in arrival delay (filled with the median inside each model).",
        f"- Duplicate rows: {quality['duplicate_rows']}.",
        f"- Departure and arrival delay correlation: {quality['delay_correlation']}.",
        "- Share of 0 ratings (read as \"not rated\"): "
        + ", ".join(f"{k} {v:.1%}" for k, v in quality["zero_ratings_share"].items())
        + ".",
        "",
        "## Model choice (validation set)",
        "",
        md(val_table[cols]),
        "",
        f"Chosen by ROC AUC: **{best}**.",
        "",
        "## Test set (used once)",
        "",
        md(test_scores[cols]),
        "",
        "## Drivers of satisfaction",
        "",
        md(imp.head(10)[["feature", "auc_drop", "cumulative_share"]]),
        "",
        "![drivers](figures/drivers.png)",
        "",
        "## Where dissatisfied passengers rate low",
        "",
        md(gaps.head(8)),
        "",
        "## Row-order check",
        "",
        md(order),
        "",
        md(profile),
        "",
        md(zeros),
        "",
        f"## Satisfaction by {top} rating",
        "",
        md(levels),
        "",
    ]
    (out / "results.md").write_text("\n".join(report), encoding="utf-8")
    print(test_scores[cols].to_string(index=False))
    print(order.to_string(index=False))


if __name__ == "__main__":
    main()
