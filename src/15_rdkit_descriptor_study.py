"""Full study with RDKit 2D molecular descriptors as the molecular
representation, mirroring the structure of 07_fixed_holdout_morgan_study.py.

Produces every number the manuscript needs:
  * model-family selection by mean five-fold training cross-validation
  * pooled out-of-fold predictions for the training set
  * fixed internal holdout evaluation
  * scaffold-grouped five-fold cross-validation
  * applicability domain from nearest-neighbour Euclidean distance in
    standardized descriptor space (the direct analogue of the Tanimoto rule:
    95th percentile of out-of-fold nearest distances, set on training folds only)
  * reference predictors (training mean, 1-NN in descriptor space)
  * SHAP attributions for the holdout

Descriptor preprocessing (median imputation, zero-variance removal) is inside
the pipeline, so it is fit on each fold's fitting portion only. Standardization
is used for the distance-based applicability domain, not for the tree models.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
NJOBS = int(os.environ.get("AUDIT_NJOBS", "1"))
MODELS = ("random_forest", "xgboost", "gradient_boosting", "extra_trees")

_spec = importlib.util.spec_from_file_location(
    "ga", str(Path(__file__).resolve().parent / "09_generalization_audit.py"))
ga = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ga)


def descriptor_block(smiles) -> pd.DataFrame:
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        rows.append(Descriptors.CalcMolDescriptors(mol) if mol is not None else {})
    X = pd.DataFrame(rows).astype(float)
    return X.replace([np.inf, -np.inf], np.nan)


def build_estimator(name: str, columns: list[str]) -> Pipeline:
    regressors = {
        "random_forest": RandomForestRegressor(random_state=42, n_jobs=NJOBS),
        "xgboost": xgb.XGBRegressor(random_state=42, n_jobs=NJOBS),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "extra_trees": ExtraTreesRegressor(random_state=42, n_jobs=NJOBS),
    }
    prep = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("variance", VarianceThreshold(0.0))])
    features = ColumnTransformer([("desc", prep, columns)], remainder="drop")
    return Pipeline([("features", features), ("reg", regressors[name])])


def search_space_for(name):
    if name in ("random_forest", "extra_trees"):
        return {"reg__n_estimators": [100, 200, 300, 500, 800],
                "reg__max_depth": [None, 8, 12, 16, 24],
                "reg__min_samples_leaf": [1, 2, 3, 5],
                "reg__min_samples_split": [2, 4, 8],
                "reg__max_features": [0.3, 0.5, 0.75, 1.0]}
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
                "reg__colsample_bytree": [0.5, 0.7, 0.85, 1.0],
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
    for cand in ParameterSampler(search_space_for(name), n_iter=n_iter,
                                 random_state=seed):
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


def cv_select(name, X, y, fit_idx, cv, n_iter, seed):
    """Return (best params, mean CV R2, fold metric means) on the fitting set."""
    best, best_r2, best_folds = None, -np.inf, None
    for cand in candidates(name, n_iter, seed):
        pipe = build_estimator(name, list(X.columns)).set_params(**cand)
        r2s, rmses, maes = [], [], []
        for itr, ite in cv.split(fit_idx):
            pipe.fit(X.iloc[fit_idx[itr]], y[fit_idx[itr]])
            pred = pipe.predict(X.iloc[fit_idx[ite]])
            r2s.append(r2_score(y[fit_idx[ite]], pred))
            rmses.append(mean_squared_error(y[fit_idx[ite]], pred) ** 0.5)
            maes.append(mean_absolute_error(y[fit_idx[ite]], pred))
        if float(np.mean(r2s)) > best_r2:
            best, best_r2 = cand, float(np.mean(r2s))
            best_folds = {"cv_r2_mean": float(np.mean(r2s)),
                          "cv_r2_sd": float(np.std(r2s, ddof=1)),
                          "cv_rmse_mean": float(np.mean(rmses)),
                          "cv_rmse_sd": float(np.std(rmses, ddof=1)),
                          "cv_mae_mean": float(np.mean(maes)),
                          "cv_mae_sd": float(np.std(maes, ddof=1))}
    return best, best_r2, best_folds


def pooled_oof(name, params, X, y, fit_idx, cv):
    oof = np.full(len(fit_idx), np.nan)
    for itr, ite in cv.split(fit_idx):
        pipe = build_estimator(name, list(X.columns)).set_params(**params)
        pipe.fit(X.iloc[fit_idx[itr]], y[fit_idx[itr]])
        oof[ite] = pipe.predict(X.iloc[fit_idx[ite]])
    return oof


def standardized(X, fit_idx):
    """Impute + drop zero-variance + standardize, fit on fit_idx only."""
    prep = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("variance", VarianceThreshold(0.0)),
                     ("scale", StandardScaler())])
    prep.fit(X.iloc[fit_idx])
    return prep.transform(X), prep


def nearest_distance(Z, query_idx, reference_idx):
    """Euclidean distance from each query row to its nearest reference row."""
    out = np.empty(len(query_idx), dtype=float)
    ref = Z[reference_idx]
    for i, q in enumerate(query_idx):
        d = np.sqrt(((ref - Z[q]) ** 2).sum(axis=1))
        out[i] = d.min()
    return out


def nn_descriptor_reference(Z, y, query_idx, reference_idx):
    ref = Z[reference_idx]
    pred = np.empty(len(query_idx), dtype=float)
    for i, q in enumerate(query_idx):
        d = ((ref - Z[q]) ** 2).sum(axis=1)
        pred[i] = y[reference_idx[int(np.argmin(d))]]
    return pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--search-iterations", type=int, default=40)
    p.add_argument("--skip-shap", action="store_true")
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.data_file)
    d = d[d["method"] == "tbars_mda"]
    d = d[d["pic50"].between(2.0, 9.5)]
    d = d[~d["outlier_flag"].astype(bool)].reset_index(drop=True)
    if d["struct_key"].duplicated().any():
        raise RuntimeError("duplicate standardized structures")
    d["scaffold"] = d["smiles"].map(ga.scaffold_key)
    y = d["pic50"].to_numpy(dtype=float)

    X = descriptor_block(d["smiles"])
    X.to_csv(out / "descriptors_378.csv", index=False, encoding="utf-8-sig")
    print(f"structures {len(d)}  descriptors {X.shape[1]}  "
          f"scaffolds {d['scaffold'].nunique()}", flush=True)
    n_missing = int(X.isna().any(axis=0).sum())
    print(f"descriptor columns with missing values: {n_missing}", flush=True)

    idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=0.2, random_state=args.seed, shuffle=True)
    tr, te = np.sort(tr), np.sort(te)
    manifest = d[["struct_key", "smiles", "pic50", "doi", "scaffold"]].copy()
    manifest.insert(0, "partition", np.where(manifest.index.isin(te), "test", "train"))
    manifest.to_csv(out / "split_manifest.csv", index=False, encoding="utf-8-sig")
    cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    # ---------------- model family selection on the training set ----------
    print("\n[1/4] model selection by training cross-validation", flush=True)
    selection, chosen = [], None
    for name in MODELS:
        best, cv_r2, folds = cv_select(name, X, y, tr, cv, args.search_iterations,
                                       args.seed)
        pipe = build_estimator(name, list(X.columns)).set_params(**best)
        pipe.fit(X.iloc[tr], y[tr])
        hold = score(y[te], pipe.predict(X.iloc[te]))
        selection.append({"model": name, **folds,
                          "selected_params": json.dumps(best, sort_keys=True),
                          "holdout_r2": hold["r2"], "holdout_rmse": hold["rmse"],
                          "holdout_mae": hold["mae"]})
        print(f"  {name:18s} CV R2={cv_r2:.4f}  holdout R2={hold['r2']:.4f} "
              f"RMSE={hold['rmse']:.4f} MAE={hold['mae']:.4f}", flush=True)
        if chosen is None or cv_r2 > chosen[1]:
            chosen = (name, cv_r2, best, folds)
    model_table = pd.DataFrame(selection)
    model_table.to_csv(out / "Table3_model_selection.csv", index=False,
                       encoding="utf-8-sig")
    name, cv_r2, params, folds = chosen
    print(f"\n  selected by mean training CV R2: {name} ({cv_r2:.4f})", flush=True)
    print(f"  parameters: {params}", flush=True)

    # ---------------- pooled OOF and holdout ------------------------------
    print("\n[2/4] pooled out-of-fold and holdout evaluation", flush=True)
    oof = pooled_oof(name, params, X, y, tr, cv)
    pooled = score(y[tr], oof)
    pipe = build_estimator(name, list(X.columns)).set_params(**params)
    pipe.fit(X.iloc[tr], y[tr])
    hold_pred = pipe.predict(X.iloc[te])
    holdout = score(y[te], hold_pred)
    in_sample = score(y[tr], pipe.predict(X.iloc[tr]))
    print(f"  pooled OOF  R2={pooled['r2']:.4f} RMSE={pooled['rmse']:.4f} "
          f"MAE={pooled['mae']:.4f}", flush=True)
    print(f"  holdout     R2={holdout['r2']:.4f} RMSE={holdout['rmse']:.4f} "
          f"MAE={holdout['mae']:.4f}", flush=True)
    print(f"  in-sample   R2={in_sample['r2']:.4f}", flush=True)
    pd.DataFrame({"struct_key": d["struct_key"].iloc[te].to_numpy(),
                  "smiles": d["smiles"].iloc[te].to_numpy(),
                  "observed_pic50": y[te], "predicted_pic50": hold_pred}).to_csv(
        out / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"struct_key": d["struct_key"].iloc[tr].to_numpy(),
                  "observed_pic50": y[tr], "oof_prediction": oof}).to_csv(
        out / "training_oof_predictions.csv", index=False, encoding="utf-8-sig")

    # ---------------- applicability domain --------------------------------
    print("\n[3/4] applicability domain (nearest-neighbour Euclidean distance)",
          flush=True)
    Z, _ = standardized(X, tr)
    fold_nearest = []
    for fold_train, fold_valid in cv.split(tr):
        fold_nearest.extend(nearest_distance(Z, tr[fold_valid], tr[fold_train]))
    cutoff = float(np.quantile(fold_nearest, 0.95))
    nearest_te = nearest_distance(Z, te, tr)
    inside = nearest_te <= cutoff
    ad = pd.DataFrame({"struct_key": d["struct_key"].iloc[te].to_numpy(),
                       "smiles": d["smiles"].iloc[te].to_numpy(),
                       "observed_pic50": y[te], "predicted_pic50": hold_pred,
                       "nearest_training_distance": nearest_te,
                       "within_domain": inside,
                       "absolute_error": np.abs(hold_pred - y[te])})
    ad.to_csv(out / "applicability_domain_holdout.csv", index=False,
              encoding="utf-8-sig")
    ad_report = {
        "method": ("nearest-neighbour Euclidean distance in standardized "
                   "descriptor space"),
        "cutoff_rule": ("95th percentile of out-of-fold nearest distances across "
                        "the five training cross-validation folds"),
        "cutoff_set_using": "302 training structures only; no holdout outcomes",
        "cutoff": cutoff,
        "training_cv_distance_median": float(np.median(fold_nearest)),
        "holdout_distance_median": float(np.median(nearest_te)),
        "inside": score(y[te][inside], hold_pred[inside]),
        "outside": score(y[te][~inside], hold_pred[~inside]) if (~inside).any()
        else {"n": 0},
    }
    print(f"  cutoff={cutoff:.4f}  inside={int(inside.sum())}/{len(te)}", flush=True)
    print(f"  inside  RMSE={ad_report['inside']['rmse']:.4f} "
          f"MAE={ad_report['inside']['mae']:.4f}", flush=True)
    if ad_report["outside"]["n"]:
        print(f"  outside RMSE={ad_report['outside']['rmse']:.4f} "
              f"MAE={ad_report['outside']['mae']:.4f}", flush=True)
    (out / "applicability_domain_report.json").write_text(
        json.dumps(ad_report, indent=2), encoding="utf-8")

    # ---------------- scaffold-grouped design + references ----------------
    print("\n[4/4] scaffold-grouped five-fold cross-validation", flush=True)
    codes = pd.factorize(d["scaffold"])[0]
    gcv = GroupKFold(n_splits=5)
    s_oof = np.full(len(y), np.nan)
    s_mean = np.full(len(y), np.nan)
    s_nn = np.full(len(y), np.nan)
    fold_rows = []
    for fold, (ftr, fte) in enumerate(gcv.split(X, y, groups=codes), start=1):
        best, inner_r2, _ = cv_select(name, X, y, ftr, cv, args.search_iterations,
                                      args.seed)
        pp = build_estimator(name, list(X.columns)).set_params(**best)
        pp.fit(X.iloc[ftr], y[ftr])
        s_oof[fte] = pp.predict(X.iloc[fte])
        s_mean[fte] = y[ftr].mean()
        Zf, _ = standardized(X, ftr)
        s_nn[fte] = nn_descriptor_reference(Zf, y, fte, ftr)
        fs = score(y[fte], s_oof[fte])
        fold_rows.append({"fold": fold, "n_fit": int(len(ftr)),
                          "n_eval": int(len(fte)), "inner_cv_r2": inner_r2,
                          **{f"eval_{k}": v for k, v in fs.items()}})
        print(f"  fold {fold}: inner={inner_r2:.4f} eval R2={fs['r2']:.4f} "
              f"RMSE={fs['rmse']:.4f}", flush=True)
    scaffold = score(y, s_oof)
    print(f"  pooled scaffold R2={scaffold['r2']:.4f} RMSE={scaffold['rmse']:.4f} "
          f"MAE={scaffold['mae']:.4f}", flush=True)
    pd.DataFrame(fold_rows).to_csv(out / "scaffold_fold_metrics.csv", index=False,
                                   encoding="utf-8-sig")
    pd.DataFrame({"struct_key": d["struct_key"], "smiles": d["smiles"],
                  "scaffold": d["scaffold"], "observed_pic50": y,
                  "oof_prediction": s_oof, "mean_reference": s_mean,
                  "nn_reference": s_nn}).to_csv(
        out / "scaffold_grouped_predictions.csv", index=False, encoding="utf-8-sig")

    # references on the random holdout
    Zr, _ = standardized(X, tr)
    hold_mean = np.full(len(te), y[tr].mean())
    hold_nn = nn_descriptor_reference(Zr, y, te, tr)

    results = {
        "dataset": {"n_structures": int(len(d)),
                    "n_measurements": int(d["n_meas"].sum()),
                    "n_scaffolds": int(d["scaffold"].nunique()),
                    "n_descriptors": int(X.shape[1]),
                    "n_descriptor_columns_with_missing": n_missing,
                    "pic50_mean": float(y.mean()), "pic50_sd": float(y.std(ddof=1)),
                    "train": {"n": int(len(tr)), "mean": float(y[tr].mean()),
                              "sd": float(y[tr].std(ddof=1))},
                    "holdout": {"n": int(len(te)), "mean": float(y[te].mean()),
                                "sd": float(y[te].std(ddof=1))}},
        "selected_model": {"family": name, "cv_r2_mean": cv_r2,
                           "params": json.loads(params and json.dumps(params)),
                           **folds},
        "random_design": {"pooled_oof": pooled, "holdout": holdout,
                          "in_sample": in_sample,
                          "mean_reference": score(y[te], hold_mean),
                          "nn_reference": score(y[te], hold_nn)},
        "scaffold_design": {"n_groups": int(len(set(codes))), "pooled": scaffold,
                            "mean_reference": score(y, s_mean),
                            "nn_reference": score(y, s_nn),
                            "fold_mean_r2": float(np.mean([r["eval_r2"] for r in fold_rows])),
                            "fold_sd_r2": float(np.std([r["eval_r2"] for r in fold_rows], ddof=1)),
                            "folds": fold_rows},
        "applicability_domain": ad_report,
        "model_selection_table": selection,
    }
    (out / "rdkit_study_report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- SHAP -------------------------------------------------
    if not args.skip_shap:
        print("\n[extra] SHAP attributions on the holdout", flush=True)
        try:
            import shap
            prep = pipe.named_steps["features"]
            Xt_tr = prep.transform(X.iloc[tr])
            Xt_te = prep.transform(X.iloc[te])
            kept = prep.named_transformers_["desc"].named_steps["variance"].get_support()
            names = [c for c, k in zip(X.columns, kept) if k]
            explainer = shap.TreeExplainer(pipe.named_steps["reg"], Xt_tr,
                                           feature_perturbation="tree_path_dependent")
            sv = explainer.shap_values(Xt_te, check_additivity=False)
            mean_abs = np.abs(sv).mean(axis=0)
            imp = (pd.DataFrame({"descriptor": names, "mean_abs_shap": mean_abs,
                                 "mean_shap": sv.mean(axis=0)})
                   .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
            imp.to_csv(out / "shap_descriptor_importance.csv", index=False,
                       encoding="utf-8-sig")
            pd.DataFrame(sv, columns=names).to_csv(
                out / "shap_values_holdout.csv", index=False, encoding="utf-8-sig")
            print(imp.head(15).round(4).to_string(index=False), flush=True)
        except Exception as exc:
            print(f"  SHAP failed: {exc}", flush=True)

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
