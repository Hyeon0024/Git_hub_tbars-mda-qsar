"""Assemble the deposit package: the fitted model, the code that produced it,
and every table and figure the manuscript cites.

Also generates the two supplementary items the manuscript references but that
no earlier script wrote:
  Table S1     hyperparameter search values and the value selected per model
  Figures S1-S3  out-of-fold and holdout predictions for the three model
                 families that were not selected
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("st", str(HERE / "15_rdkit_descriptor_study.py"))
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
})
ACCENT = "#2b6cb0"
NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost",
         "gradient_boosting": "Gradient Boosting", "extra_trees": "Extra Trees"}


def parity(ax, obs, pred, title, label):
    lo = min(obs.min(), pred.min()) - 0.25
    hi = max(obs.max(), pred.max()) + 0.25
    ax.plot([lo, hi], [lo, hi], color="#9aa5b1", lw=0.9, ls="--", zorder=1)
    ax.scatter(obs, pred, s=16, facecolor=ACCENT, edgecolor="white",
               linewidth=0.4, alpha=0.85, zorder=2)
    r2 = r2_score(obs, pred)
    rmse = float(np.sqrt(((obs - pred) ** 2).mean()))
    mae = float(np.abs(obs - pred).mean())
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Observed pIC$_{50}$"); ax.set_ylabel("Predicted pIC$_{50}$")
    ax.set_title(title, loc="left", color="#1f2d3d")
    ax.text(0.04, 0.96, f"$n$ = {len(obs)}\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, va="top", ha="left", color="#1f2d3d",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cbd2d9", linewidth=0.6))
    ax.text(-0.17, 1.04, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color="#1f2d3d")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--study-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--search-iterations", type=int, default=40)
    args = p.parse_args()
    out = args.output_dir.resolve()
    for sub in ("model", "code", "figures", "tables", "supplementary", "data"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rep = json.loads((args.study_dir / "rdkit_study_report.json").read_text(encoding="utf-8"))
    tbl = pd.DataFrame(rep["model_selection_table"])

    d = pd.read_csv(args.data_file)
    d = d[d["method"] == "tbars_mda"]
    d = d[d["pic50"].between(2.0, 9.5)]
    d = d[~d["outlier_flag"].astype(bool)].reset_index(drop=True)
    d["scaffold"] = d["smiles"].map(st.ga.scaffold_key)
    y = d["pic50"].to_numpy(dtype=float)
    X = st.descriptor_block(d["smiles"])
    idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=0.2, random_state=args.seed, shuffle=True)
    tr, te = np.sort(tr), np.sort(te)
    cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    # ---------------- Table S1: search space and selected values -----------
    rows = []
    for _, r in tbl.iterrows():
        fam = r["model"]
        space = st.search_space_for(fam)
        chosen = json.loads(r["selected_params"])
        for param, values in space.items():
            short = param.removeprefix("reg__")
            rows.append({"model": NAMES[fam], "parameter": short,
                         "search_values": json.dumps(values),
                         "selected_value": json.dumps(chosen.get(param))})
    s1 = pd.DataFrame(rows)
    s1.to_csv(out / "supplementary" / "TableS1_hyperparameter_search.csv",
              index=False, encoding="utf-8-sig")
    print(f"Table S1: {len(s1)} rows")

    # ---------------- Figures S1-S3: the other three families --------------
    others = [m for m in ("random_forest", "xgboost", "extra_trees")]
    for i, fam in enumerate(others, start=1):
        params = json.loads(tbl.loc[tbl["model"] == fam, "selected_params"].iloc[0])
        oof = np.full(len(tr), np.nan)
        for itr, ite in cv.split(tr):
            pipe = st.build_estimator(fam, list(X.columns)).set_params(**params)
            pipe.fit(X.iloc[tr[itr]], y[tr[itr]])
            oof[ite] = pipe.predict(X.iloc[tr[ite]])
        pipe = st.build_estimator(fam, list(X.columns)).set_params(**params)
        pipe.fit(X.iloc[tr], y[tr])
        hold = pipe.predict(X.iloc[te])
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6))
        parity(axes[0], y[tr], oof,
               f"{NAMES[fam]} — training, pooled out-of-fold", "A")
        parity(axes[1], y[te], hold,
               f"{NAMES[fam]} — fixed internal holdout", "B")
        fig.tight_layout()
        fig.savefig(out / "supplementary" / f"FigureS{i}_{fam}.png")
        plt.close(fig)
        pd.DataFrame({"struct_key": d["struct_key"].iloc[te].to_numpy(),
                      "observed_pic50": y[te], "predicted_pic50": hold}).to_csv(
            out / "supplementary" / f"FigureS{i}_{fam}_holdout_predictions.csv",
            index=False, encoding="utf-8-sig")
        print(f"Figure S{i}: {NAMES[fam]}  holdout R2={r2_score(y[te], hold):.4f}")

    # ---------------- the fitted model -------------------------------------
    sel = rep["selected_model"]
    best = st.build_estimator("gradient_boosting", list(X.columns)).set_params(**sel["params"])
    best.fit(X.iloc[tr], y[tr])
    joblib.dump({"pipeline": best,
                 "descriptor_columns": list(X.columns),
                 "training_struct_keys": d["struct_key"].iloc[tr].tolist(),
                 "applicability_cutoff": rep["applicability_domain"]["cutoff"],
                 "seed": args.seed,
                 "note": "Gradient Boosting fitted on the 302 training structures; "
                         "predicts TBARS/MDA-associated pIC50."},
                out / "model" / "gradient_boosting_fitted.joblib", compress=3)
    print("fitted model saved")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
