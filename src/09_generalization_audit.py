"""Phase 9: generalization audit requested in pre-submission review.

Adds, on top of the single random 80:20 split used in the manuscript:
  (a) Bemis-Murcko scaffold-grouped validation
  (b) DOI (source-publication) grouped validation
  (c) nested cross-validation that includes fingerprint, hyperparameter
      and model-family selection in the inner loop
  (d) trivial reference models (training mean, 1-NN Tanimoto)

Reuses the estimator/pipeline definitions of 07_fixed_holdout_morgan_study.py
so the comparison is like-for-like.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

RDLogger.DisableLog("rdApp.*")

MODELS = ("random_forest", "xgboost", "gradient_boosting", "extra_trees")
FP_GRID = [(r, b) for r in (1, 2, 3) for b in (1024, 2048, 4096)]
_NJOBS = int(__import__("os").environ.get("AUDIT_NJOBS", "1"))


class MinFrequencyFilter(BaseEstimator, TransformerMixin):
    """Retain bits present in at least `min_count` training-fold molecules."""

    def __init__(self, min_count: int = 5):
        self.min_count = min_count

    def fit(self, X, y=None):
        self.keep_ = np.asarray((np.asarray(X) != 0).sum(axis=0) >= self.min_count).ravel()
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def morgan_fingerprints(smiles, radius=2, n_bits=2048) -> pd.DataFrame:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mat = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), mat[i])
    return pd.DataFrame(mat, columns=[f"morgan_{j}" for j in range(n_bits)])


def build_estimator(name: str, columns: list[str]) -> Pipeline:
    regressors = {
        "random_forest": RandomForestRegressor(random_state=42, n_jobs=_NJOBS),
        "xgboost": xgb.XGBRegressor(random_state=42, n_jobs=_NJOBS),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "extra_trees": ExtraTreesRegressor(random_state=42, n_jobs=_NJOBS),
    }
    features = ColumnTransformer([("fp", MinFrequencyFilter(5), columns)], remainder="drop")
    return Pipeline([("features", features), ("reg", regressors[name])])


def search_space_for(name):
    if name in ("random_forest", "extra_trees"):
        return {"reg__n_estimators": [100, 200, 300, 500, 800],
                "reg__max_depth": [None, 8, 12, 16, 24],
                "reg__min_samples_leaf": [1, 2, 3, 5],
                "reg__min_samples_split": [2, 4, 8],
                "reg__max_features": [0.5, 0.75, 1.0]}
    if name == "gradient_boosting":
        return {"reg__n_estimators": [100, 200, 300, 500, 800],
                "reg__learning_rate": [0.02, 0.05, 0.075, 0.1, 0.15],
                "reg__max_depth": [2, 3, 4, 5],
                "reg__subsample": [0.7, 0.85, 1.0],
                "reg__min_samples_leaf": [1, 3, 5]}
    if name == "xgboost":
        return {"reg__n_estimators": [100, 200, 300, 500, 800],
                "reg__learning_rate": [0.02, 0.05, 0.075, 0.1, 0.15],
                "reg__max_depth": [2, 3, 4, 5, 7],
                "reg__subsample": [0.7, 0.85, 1.0],
                "reg__colsample_bytree": [0.7, 0.85, 1.0],
                "reg__min_child_weight": [1, 3, 5]}
    raise ValueError(name)


def baseline_for(name):
    if name in ("random_forest", "extra_trees"):
        return {"reg__n_estimators": 200, "reg__max_depth": None,
                "reg__min_samples_leaf": 1 if name == "random_forest" else 3,
                "reg__min_samples_split": 2, "reg__max_features": 1.0}
    if name == "gradient_boosting":
        return {"reg__n_estimators": 200, "reg__learning_rate": 0.1,
                "reg__max_depth": 5, "reg__subsample": 1.0, "reg__min_samples_leaf": 1}
    if name == "xgboost":
        return {"reg__n_estimators": 200, "reg__learning_rate": 0.1,
                "reg__max_depth": 5, "reg__subsample": 1.0,
                "reg__colsample_bytree": 1.0, "reg__min_child_weight": 1}
    raise ValueError(name)


def candidates(name, n_iter, seed):
    base = baseline_for(name)
    sampled = []
    for cand in ParameterSampler(search_space_for(name), n_iter=n_iter, random_state=seed):
        if cand != base:
            sampled.append(cand)
        if len(sampled) == n_iter - 1:
            break
    return [*sampled, base]


def score(y, pred):
    return {"n": int(len(y)),
            "r2": float(r2_score(y, pred)) if len(y) >= 2 else None,
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "mae": float(mean_absolute_error(y, pred))}


# --------------------------------------------------------------------------
# grouping keys
# --------------------------------------------------------------------------

def scaffold_key(smi: str) -> str:
    """Bemis-Murcko generic scaffold; falls back to the molecule itself."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        key = Chem.MolToSmiles(generic)
    except Exception:
        key = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return key if key else "acyclic"


# --------------------------------------------------------------------------
# reference (trivial) models
# --------------------------------------------------------------------------

def mean_reference(y_train, n_test):
    return np.full(n_test, float(np.mean(y_train)))


def nn_reference(fps_train, y_train, fps_test):
    """1-nearest-neighbour Tanimoto prediction."""
    out = np.empty(len(fps_test), dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    for i, fp in enumerate(fps_test):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, list(fps_train)))
        out[i] = y_train[int(np.argmax(sims))]
    return out


def rdkit_fps(smiles, radius, n_bits):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


# --------------------------------------------------------------------------
# evaluation routines
# --------------------------------------------------------------------------

def grouped_evaluation(data, groups, label, radius, bits, model_name, seed, jobs, n_iter):
    """Group-wise K-fold: no group appears in both fit and evaluation sets."""
    X = morgan_fingerprints(data["smiles"], radius, bits)
    y = data["pic50"].to_numpy(dtype=float)
    codes = pd.factorize(groups)[0]
    n_groups = len(set(codes))
    n_splits = min(5, n_groups)
    cv = GroupKFold(n_splits=n_splits)

    fps = rdkit_fps(data["smiles"], radius, bits)
    oof = np.full(len(y), np.nan)
    oof_mean = np.full(len(y), np.nan)
    oof_nn = np.full(len(y), np.nan)
    fold_rows = []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=codes), start=1):
        inner = KFold(n_splits=5, shuffle=True, random_state=seed)
        best, best_r2 = None, -np.inf
        for cand in candidates(model_name, n_iter, seed):
            pipe = build_estimator(model_name, list(X.columns)).set_params(**cand)
            fold_scores = []
            for itr, ite in inner.split(tr):
                pipe.fit(X.iloc[tr[itr]], y[tr[itr]])
                fold_scores.append(r2_score(y[tr[ite]], pipe.predict(X.iloc[tr[ite]])))
            mean_r2 = float(np.mean(fold_scores))
            if mean_r2 > best_r2:
                best, best_r2 = cand, mean_r2
        pipe = build_estimator(model_name, list(X.columns)).set_params(**best)
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict(X.iloc[te])
        oof_mean[te] = mean_reference(y[tr], len(te))
        oof_nn[te] = nn_reference([fps[i] for i in tr], y[tr], [fps[i] for i in te])
        fold_rows.append({"fold": fold, "n_fit": int(len(tr)), "n_eval": int(len(te)),
                          "inner_best_r2": best_r2,
                          "selected_params": json.dumps(best, sort_keys=True),
                          **{f"eval_{k}": v for k, v in score(y[te], oof[te]).items()}})

    return {
        "scheme": label,
        "n_groups": int(n_groups),
        "n_splits": int(n_splits),
        "model": model_name,
        "radius": radius,
        "bits": bits,
        "pooled_model": score(y, oof),
        "pooled_mean_reference": score(y, oof_mean),
        "pooled_nn_reference": score(y, oof_nn),
        "fold_mean_r2": float(np.mean([r["eval_r2"] for r in fold_rows
                                       if r["eval_r2"] is not None])),
        "fold_sd_r2": float(np.std([r["eval_r2"] for r in fold_rows
                                    if r["eval_r2"] is not None], ddof=1)),
        "folds": fold_rows,
    }, oof, oof_mean, oof_nn


def nested_cv(data, seed, jobs, n_iter, outer_splits=5):
    """Outer loop estimates performance; inner loop selects fp, params and family."""
    y = data["pic50"].to_numpy(dtype=float)
    fp_cache = {(r, b): morgan_fingerprints(data["smiles"], r, b) for r, b in FP_GRID}
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    rows = []

    for fold, (tr, te) in enumerate(outer.split(y), start=1):
        inner = KFold(n_splits=5, shuffle=True, random_state=seed)
        # stage 1: fingerprint setting, screened with the 120-tree RF of the manuscript
        best_fp, best_fp_r2 = None, -np.inf
        for key in FP_GRID:
            X = fp_cache[key]
            pipe = build_estimator("random_forest", list(X.columns)).set_params(
                reg__n_estimators=120, reg__max_depth=None, reg__min_samples_leaf=1)
            s = []
            for itr, ite in inner.split(tr):
                pipe.fit(X.iloc[tr[itr]], y[tr[itr]])
                s.append(r2_score(y[tr[ite]], pipe.predict(X.iloc[tr[ite]])))
            if float(np.mean(s)) > best_fp_r2:
                best_fp, best_fp_r2 = key, float(np.mean(s))
        X = fp_cache[best_fp]
        # stage 2: model family and hyperparameters
        best = None
        for name in MODELS:
            for cand in candidates(name, n_iter, seed):
                pipe = build_estimator(name, list(X.columns)).set_params(**cand)
                s = []
                for itr, ite in inner.split(tr):
                    pipe.fit(X.iloc[tr[itr]], y[tr[itr]])
                    s.append(r2_score(y[tr[ite]], pipe.predict(X.iloc[tr[ite]])))
                mean_r2 = float(np.mean(s))
                if best is None or mean_r2 > best[0]:
                    best = (mean_r2, name, cand)
        inner_r2, name, cand = best
        pipe = build_estimator(name, list(X.columns)).set_params(**cand)
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict(X.iloc[te])
        rows.append({"fold": fold, "n_fit": int(len(tr)), "n_eval": int(len(te)),
                     "selected_radius": best_fp[0], "selected_bits": best_fp[1],
                     "selected_model": name, "inner_best_r2": inner_r2,
                     "selected_params": json.dumps(cand, sort_keys=True),
                     **{f"eval_{k}": v for k, v in score(y[te], oof[te]).items()}})
        print(f"  nested fold {fold}: fp={best_fp} model={name} "
              f"inner_r2={inner_r2:.4f} outer_r2={rows[-1]['eval_r2']:.4f}", flush=True)

    return {
        "scheme": "nested_cv_with_model_selection",
        "outer_splits": outer_splits,
        "pooled": score(y, oof),
        "fold_mean_r2": float(np.mean([r["eval_r2"] for r in rows])),
        "fold_sd_r2": float(np.std([r["eval_r2"] for r in rows], ddof=1)),
        "folds": rows,
    }, oof


def random_holdout_reference(data, seed, radius, bits, model_name, n_iter):
    """Reproduce the manuscript's single random split and add trivial references."""
    X = morgan_fingerprints(data["smiles"], radius, bits)
    y = data["pic50"].to_numpy(dtype=float)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)
    # 07_fixed_holdout_morgan_study.py sorts both partitions after splitting;
    # the inner CV folds depend on this ordering, so it must be reproduced here
    # or hyperparameter selection diverges from the manuscript.
    tr, te = np.sort(tr), np.sort(te)
    inner = KFold(n_splits=5, shuffle=True, random_state=seed)

    best, best_r2 = None, -np.inf
    for cand in candidates(model_name, n_iter, seed):
        pipe = build_estimator(model_name, list(X.columns)).set_params(**cand)
        s = []
        for itr, ite in inner.split(tr):
            pipe.fit(X.iloc[tr[itr]], y[tr[itr]])
            s.append(r2_score(y[tr[ite]], pipe.predict(X.iloc[tr[ite]])))
        if float(np.mean(s)) > best_r2:
            best, best_r2 = cand, float(np.mean(s))

    pipe = build_estimator(model_name, list(X.columns)).set_params(**best)
    pipe.fit(X.iloc[tr], y[tr])
    pred = pipe.predict(X.iloc[te])
    fps = rdkit_fps(data["smiles"], radius, bits)
    return {
        "scheme": "single_random_split_80_20",
        "model": model_name,
        "cv_mean_r2_training": best_r2,
        "selected_params": json.dumps(best, sort_keys=True),
        "holdout_model": score(y[te], pred),
        "holdout_mean_reference": score(y[te], mean_reference(y[tr], len(te))),
        "holdout_nn_reference": score(
            y[te], nn_reference([fps[i] for i in tr], y[tr], [fps[i] for i in te])),
        "n_train": int(len(tr)), "n_holdout": int(len(te)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method-filter", default="tbars_mda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--search-iterations", type=int, default=40)
    p.add_argument("--model", default="random_forest")
    args = p.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data_file)
    if args.method_filter:
        data = data[data["method"] == args.method_filter].copy()
    data = data[data["pic50"].between(2.0, 9.5)].copy()
    data = data[~data["outlier_flag"].astype(bool)].reset_index(drop=True)
    print(f"structures: {len(data)}  measurements: {int(data['n_meas'].sum())}  "
          f"DOIs: {data['doi'].nunique()}", flush=True)

    data["scaffold"] = data["smiles"].map(scaffold_key)
    print(f"generic Bemis-Murcko scaffolds: {data['scaffold'].nunique()}", flush=True)
    data[["struct_key", "smiles", "pic50", "doi", "scaffold", "tissue", "inducer",
          "n_meas"]].to_csv(out / "grouping_keys.csv", index=False, encoding="utf-8-sig")

    results = {}

    print("\n[1/4] single random split (manuscript design) + trivial references", flush=True)
    results["random_split"] = random_holdout_reference(
        data, args.seed, 2, 2048, args.model, args.search_iterations)
    print(json.dumps(results["random_split"], indent=2), flush=True)

    print("\n[2/4] scaffold-grouped validation", flush=True)
    rep, oof, om, onn = grouped_evaluation(
        data, data["scaffold"], "scaffold_grouped_kfold", 2, 2048,
        args.model, args.seed, args.jobs, args.search_iterations)
    results["scaffold_grouped"] = rep
    pd.DataFrame({"struct_key": data["struct_key"], "smiles": data["smiles"],
                  "scaffold": data["scaffold"], "observed": data["pic50"],
                  "predicted": oof, "mean_reference": om, "nn_reference": onn}).to_csv(
        out / "scaffold_grouped_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: v for k, v in rep.items() if k != "folds"}, indent=2), flush=True)

    print("\n[3/4] DOI(source)-grouped validation", flush=True)
    rep, oof, om, onn = grouped_evaluation(
        data, data["doi"], "doi_grouped_kfold", 2, 2048,
        args.model, args.seed, args.jobs, args.search_iterations)
    results["doi_grouped"] = rep
    pd.DataFrame({"struct_key": data["struct_key"], "smiles": data["smiles"],
                  "doi": data["doi"], "observed": data["pic50"],
                  "predicted": oof, "mean_reference": om, "nn_reference": onn}).to_csv(
        out / "doi_grouped_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: v for k, v in rep.items() if k != "folds"}, indent=2), flush=True)

    print("\n[4/4] nested cross-validation including model selection", flush=True)
    rep, oof = nested_cv(data, args.seed, args.jobs, args.search_iterations)
    results["nested_cv"] = rep
    pd.DataFrame({"struct_key": data["struct_key"], "smiles": data["smiles"],
                  "observed": data["pic50"], "predicted": oof}).to_csv(
        out / "nested_cv_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: v for k, v in rep.items() if k != "folds"}, indent=2), flush=True)

    (out / "generalization_audit.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    rs = results["random_split"]
    rows.append({"scheme": "Single random 80:20 split (manuscript)",
                 "n_eval": rs["holdout_model"]["n"],
                 "model_r2": rs["holdout_model"]["r2"],
                 "model_rmse": rs["holdout_model"]["rmse"],
                 "model_mae": rs["holdout_model"]["mae"],
                 "mean_ref_rmse": rs["holdout_mean_reference"]["rmse"],
                 "nn_ref_rmse": rs["holdout_nn_reference"]["rmse"]})
    for key, label in (("scaffold_grouped", "Scaffold-grouped 5-fold"),
                       ("doi_grouped", "Source(DOI)-grouped 5-fold")):
        r = results[key]
        rows.append({"scheme": label,
                     "n_eval": r["pooled_model"]["n"],
                     "model_r2": r["pooled_model"]["r2"],
                     "model_rmse": r["pooled_model"]["rmse"],
                     "model_mae": r["pooled_model"]["mae"],
                     "mean_ref_rmse": r["pooled_mean_reference"]["rmse"],
                     "nn_ref_rmse": r["pooled_nn_reference"]["rmse"]})
    r = results["nested_cv"]
    rows.append({"scheme": "Nested CV (model selection included)",
                 "n_eval": r["pooled"]["n"], "model_r2": r["pooled"]["r2"],
                 "model_rmse": r["pooled"]["rmse"], "model_mae": r["pooled"]["mae"],
                 "mean_ref_rmse": None, "nn_ref_rmse": None})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "generalization_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
