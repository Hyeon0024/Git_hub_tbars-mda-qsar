"""Predict the human-LDL literature candidates with the descriptor model and
place each one in the distance-based applicability domain.

Evaluates every candidate supplied, not a subset chosen after inspecting the
errors, so the set is defined before the comparison is made.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
NJOBS = int(os.environ.get("AUDIT_NJOBS", "1"))

_spec = importlib.util.spec_from_file_location(
    "st", str(Path(__file__).resolve().parent / "15_rdkit_descriptor_study.py"))
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--study-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rep = json.loads(args.study_report.read_text(encoding="utf-8"))
    params = rep["selected_model"]["params"]
    cutoff = rep["applicability_domain"]["cutoff"]

    d = pd.read_csv(args.data_file)
    d = d[d["method"] == "tbars_mda"]
    d = d[d["pic50"].between(2.0, 9.5)]
    d = d[~d["outlier_flag"].astype(bool)].reset_index(drop=True)
    y = d["pic50"].to_numpy(dtype=float)
    X = st.descriptor_block(d["smiles"])
    idx = np.arange(len(d))
    tr, _ = train_test_split(idx, test_size=0.2, random_state=args.seed, shuffle=True)
    tr = np.sort(tr)

    pipe = st.build_estimator("gradient_boosting", list(X.columns)).set_params(**params)
    pipe.fit(X.iloc[tr], y[tr])
    print(f"model fitted on {len(tr)} training structures", flush=True)

    cand = pd.read_csv(args.candidates).dropna(subset=["compound"]).reset_index(drop=True)
    from rdkit.Chem.MolStandardize import rdMolStandardize
    chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()

    keys, parents = [], []
    for smi in cand["input_smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            keys.append(None); parents.append(None); continue
        par = uncharger.uncharge(chooser.choose(mol))
        parents.append(Chem.MolToSmiles(par))
        keys.append(Chem.MolToSmiles(par, isomericSmiles=False))
    cand["struct_key"] = keys
    cand["parent_smiles"] = parents
    cand["in_curated_dataset"] = cand["struct_key"].isin(set(d["struct_key"]))

    Xc = st.descriptor_block(cand["parent_smiles"])
    Xc = Xc.reindex(columns=X.columns)
    cand["predicted_pic50"] = pipe.predict(Xc)
    cand["predicted_ic50_uM"] = 10 ** (6 - cand["predicted_pic50"])
    cand["observed_pic50"] = 6 - np.log10(cand["observed_ic50_uM"].astype(float))
    cand["absolute_error_pic50"] = (cand["predicted_pic50"]
                                    - cand["observed_pic50"]).abs()
    cand["fold_error_ic50"] = 10 ** cand["absolute_error_pic50"]

    # distance to the training set in the same standardized descriptor space
    prep = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("variance", VarianceThreshold(0.0)),
                     ("scale", StandardScaler())])
    prep.fit(X.iloc[tr])
    Ztr = prep.transform(X.iloc[tr])
    Zc = prep.transform(Xc)
    dist = np.array([np.sqrt(((Ztr - z) ** 2).sum(axis=1)).min() for z in Zc])
    cand["nearest_training_distance"] = dist
    cand["within_domain"] = dist <= cutoff
    nearest_idx = [int(np.argmin(np.sqrt(((Ztr - z) ** 2).sum(axis=1)))) for z in Zc]
    cand["nearest_training_smiles"] = [d["smiles"].iloc[tr[i]] for i in nearest_idx]
    cand["nearest_training_pic50"] = [float(y[tr[i]]) for i in nearest_idx]

    cols = ["compound", "source_doi", "assay", "in_curated_dataset",
            "observed_ic50_uM", "observed_ic50_sd_uM", "observed_pic50",
            "predicted_ic50_uM", "predicted_pic50", "absolute_error_pic50",
            "fold_error_ic50", "nearest_training_distance", "within_domain",
            "nearest_training_pic50", "nearest_training_smiles",
            "struct_key", "parent_smiles"]
    cand[cols].to_csv(out / "ldl_candidate_predictions.csv", index=False,
                      encoding="utf-8-sig")

    obs = cand["observed_pic50"].to_numpy()
    pred = cand["predicted_pic50"].to_numpy()
    summary = {
        "n_candidates": int(len(cand)),
        "n_in_curated_dataset": int(cand["in_curated_dataset"].sum()),
        "n_source_publications": int(cand["source_doi"].nunique()),
        "applicability_cutoff": cutoff,
        "n_within_domain": int(cand["within_domain"].sum()),
        "all_candidates": {
            "n": int(len(cand)),
            "r2": float(r2_score(obs, pred)) if len(cand) > 1 else None,
            "rmse": float(mean_squared_error(obs, pred) ** 0.5),
            "mae": float(mean_absolute_error(obs, pred)),
            "median_fold_error": float(np.median(cand["fold_error_ic50"])),
        },
    }
    inside = cand["within_domain"].to_numpy()
    if inside.any():
        summary["within_domain_only"] = {
            "n": int(inside.sum()),
            "rmse": float(mean_squared_error(obs[inside], pred[inside]) ** 0.5),
            "mae": float(mean_absolute_error(obs[inside], pred[inside])),
        }
    if (~inside).any():
        summary["outside_domain_only"] = {
            "n": int((~inside).sum()),
            "rmse": float(mean_squared_error(obs[~inside], pred[~inside]) ** 0.5),
            "mae": float(mean_absolute_error(obs[~inside], pred[~inside])),
        }
    (out / "ldl_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    show = ["compound", "observed_ic50_uM", "predicted_ic50_uM",
            "absolute_error_pic50", "fold_error_ic50",
            "nearest_training_distance", "within_domain"]
    print()
    print(cand[show].round(3).to_string(index=False))
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
