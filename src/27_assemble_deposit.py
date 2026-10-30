"""Collect the deposit tree: only the items the manuscript actually uses.

Main-text Tables 2-4 are rebuilt from rdkit_study_report.json rather than
copied, because out/exact/Table4_validation_designs.csv is the earlier
Morgan-fingerprint table and no longer matches the manuscript. Supplementary
tables are renumbered S1-S5 so the series has no gap.

Table numbering follows the manuscript:
  Table 1  curated dataset characteristics
  Table 2  five-fold cross-validation and internal holdout, four model families
  Table 3  selected model and two reference predictors, both designs
  Table 4  published human-LDL IC50 values and model predictions
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

W = Path("/home/claude/work")
OUT = W / "out" / "deposit"
REPORT = json.loads((W / "out/rdkit/rdkit_study_report.json").read_text(encoding="utf-8"))
NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost",
         "gradient_boosting": "Gradient Boosting", "extra_trees": "Extra Trees"}
LDL_THREE = ("osajin", "flemingsin", "myricitrin")

FILES: dict[str, Path] = {
    "data/curated_structures_378.csv":   W / "out/exact/tbars_mda.csv",
    "data/curated_measurements_443.csv": W / "out/exact/tbars_mda_measurements.csv",
    "data/rdkit_descriptors_378.csv":    W / "out/rdkit/descriptors_378.csv",
    "data/split_manifest.csv":           W / "out/rdkit/split_manifest.csv",

    "predictions/holdout_predictions.csv":          W / "out/rdkit/holdout_predictions.csv",
    "predictions/training_oof_predictions.csv":     W / "out/rdkit/training_oof_predictions.csv",
    "predictions/scaffold_grouped_predictions.csv": W / "out/rdkit/scaffold_grouped_predictions.csv",
    "predictions/scaffold_fold_metrics.csv":        W / "out/rdkit/scaffold_fold_metrics.csv",
    "predictions/applicability_domain_holdout.csv": W / "out/rdkit/applicability_domain_holdout.csv",
    "predictions/external_set_predictions.csv":     W / "out/rdkit/external/external_missed_predictions.csv",

    "shap/shap_descriptor_importance.csv": W / "out/rdkit/shap_descriptor_importance.csv",
    "shap/shap_values_holdout.csv":        W / "out/rdkit/shap_values_holdout.csv",

    "figures/Figure1_observed_vs_predicted.png": W / "out/rdkit/figures/Figure1_observed_vs_predicted.png",
    "figures/Figure2_applicability_domain.png":  W / "out/rdkit/figures/Figure2_applicability_domain.png",
    "figures/Figure3_shap_importance.png":       W / "out/rdkit/figures/Figure3_shap_importance.png",
    "figures/Figure4_shap_beeswarm.png":         W / "out/rdkit/figures/Figure4_shap_beeswarm.png",

    "supplementary/TableS2_protocol_audit_443.csv": W / "out/exact/TableS3_protocol_audit_443.csv",
    "supplementary/TableS3_assay_descriptions.csv": W / "out/exact/TableS3b_description_summary.csv",
    "supplementary/TableS4_multi_measurement.csv":  W / "out/exact/TableS3c_multi_measurement_conditions.csv",
    "supplementary/TableS5_external_validation.csv": W / "out/rdkit/external/TableS4_external_validation.csv",

    "reports/rdkit_study_report.json":     W / "out/rdkit/rdkit_study_report.json",
    "reports/protocol_audit_summary.json": W / "out/exact/protocol_audit_summary.json",
    "reports/external_set_summary.json":   W / "out/rdkit/external/external_missed_summary.json",
}

CODE = [
    "01_build_dataset.py",
    "09_generalization_audit.py",
    "11_protocol_audit.py",
    "15_rdkit_descriptor_study.py",
    "16_rdkit_figures.py",
    "20_ldl_candidates.py",
    "21_external_missed_set.py",
    "26_build_deposit.py",
    "27_assemble_deposit.py",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_table1() -> pd.DataFrame:
    ds = REPORT["dataset"]
    rows = [
        ("Curated structures", ds["n_structures"]),
        ("Underlying measurements", ds["n_measurements"]),
        ("Distinct generic Bemis-Murcko scaffolds", ds["n_scaffolds"]),
        ("RDKit 2D descriptors computed", ds["n_descriptors"]),
        ("Descriptor columns with missing values", ds["n_descriptor_columns_with_missing"]),
        ("pIC50, all structures: mean", round(ds["pic50_mean"], 3)),
        ("pIC50, all structures: SD", round(ds["pic50_sd"], 3)),
        ("Training set: n", ds["train"]["n"]),
        ("Training set: pIC50 mean", round(ds["train"]["mean"], 3)),
        ("Training set: pIC50 SD", round(ds["train"]["sd"], 3)),
        ("Holdout set: n", ds["holdout"]["n"]),
        ("Holdout set: pIC50 mean", round(ds["holdout"]["mean"], 3)),
        ("Holdout set: pIC50 SD", round(ds["holdout"]["sd"], 3)),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def build_table2() -> pd.DataFrame:
    t = pd.DataFrame(REPORT["model_selection_table"]).copy()
    t["model"] = t["model"].map(NAMES)
    keep = ["model", "cv_r2_mean", "cv_r2_sd", "cv_rmse_mean", "cv_rmse_sd",
            "cv_mae_mean", "cv_mae_sd", "holdout_r2", "holdout_rmse",
            "holdout_mae", "selected_params"]
    t = t[keep]
    for c in keep[1:-1]:
        t[c] = t[c].astype(float).round(4)
    t["selected_by_training_cv"] = t["model"] == NAMES[REPORT["selected_model"]["family"]]
    return t


def build_table3() -> pd.DataFrame:
    rd, sc = REPORT["random_design"], REPORT["scaffold_design"]
    gb = "Gradient Boosting"
    nn = "Nearest training structure (1-NN, descriptor distance)"
    mn = "Training-set mean"
    rows = []
    for design, block, key in (
        ("Random 80:20 holdout (Section 2.3)", rd, "holdout"),
        ("Scaffold-grouped five-fold (Section 2.4)", sc, "pooled"),
    ):
        for label, part in ((gb, block[key]), (nn, block["nn_reference"]),
                            (mn, block["mean_reference"])):
            rows.append({"validation_design": design, "n_evaluated": part["n"],
                         "predictor": label, "r2": round(part["r2"], 4),
                         "rmse": round(part["rmse"], 4), "mae": round(part["mae"], 4)})
    return pd.DataFrame(rows)


def build_table4() -> pd.DataFrame | None:
    src = W / "out/rdkit/ldl16/ldl_candidate_predictions.csv"
    if not src.exists():
        return None
    d = pd.read_csv(src)
    name = d.columns[0]
    sel = d[d[name].str.lower().str.strip().isin(LDL_THREE)].copy()
    order = {c: i for i, c in enumerate(LDL_THREE)}
    sel = sel.sort_values(name, key=lambda s: s.str.lower().str.strip().map(order))
    return sel.reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing, copied = [], []

    # stage the items 26_build_deposit.py wrote directly into OUT/supplementary
    sup = OUT / "supplementary"
    stage = OUT / ".stage"
    stage.mkdir(exist_ok=True)
    for name in ("TableS1_hyperparameter_search.csv", "FigureS1_random_forest.png",
                 "FigureS2_xgboost.png", "FigureS3_extra_trees.png"):
        p = sup / name
        if p.exists():
            shutil.copy2(p, stage / name)
        else:
            missing.append((f"supplementary/{name}", p))
    for p in sup.glob("FigureS*_holdout_predictions.csv"):
        shutil.copy2(p, stage / p.name)
    mdl = OUT / "model/gradient_boosting_fitted.joblib"
    if mdl.exists():
        shutil.copy2(mdl, stage / mdl.name)
    else:
        missing.append(("model/gradient_boosting_fitted.joblib", mdl))

    for d in ("data", "predictions", "shap", "figures", "tables",
              "supplementary", "reports", "code", "model"):
        shutil.rmtree(OUT / d, ignore_errors=True)
        (OUT / d).mkdir(parents=True, exist_ok=True)

    for dest, src in FILES.items():
        if not src.exists():
            missing.append((dest, src))
            continue
        shutil.copy2(src, OUT / dest)
        copied.append(dest)

    for name in ("TableS1_hyperparameter_search.csv", "FigureS1_random_forest.png",
                 "FigureS2_xgboost.png", "FigureS3_extra_trees.png"):
        p = stage / name
        if p.exists():
            shutil.copy2(p, sup / name)
            copied.append(f"supplementary/{name}")
    for p in stage.glob("FigureS*_holdout_predictions.csv"):
        shutil.copy2(p, OUT / "predictions" / p.name)
        copied.append(f"predictions/{p.name}")

    stash = stage / "gradient_boosting_fitted.joblib"
    if stash.exists():
        shutil.copy2(stash, OUT / "model/gradient_boosting_fitted.joblib")
        copied.append("model/gradient_boosting_fitted.joblib")

    t1, t2, t3 = build_table1(), build_table2(), build_table3()
    t1.to_csv(OUT / "tables/Table1_dataset_characteristics.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(OUT / "tables/Table2_model_selection.csv", index=False, encoding="utf-8-sig")
    t3.to_csv(OUT / "tables/Table3_validation_designs.csv", index=False, encoding="utf-8-sig")
    copied += ["tables/Table1_dataset_characteristics.csv",
               "tables/Table2_model_selection.csv",
               "tables/Table3_validation_designs.csv"]
    t4 = build_table4()
    if t4 is not None:
        t4.to_csv(OUT / "tables/Table4_human_ldl_predictions.csv", index=False, encoding="utf-8-sig")
        copied.append("tables/Table4_human_ldl_predictions.csv")
        print(f"Table 4: {len(t4)} compounds -> {t4.iloc[:, 0].tolist()}")
    else:
        missing.append(("tables/Table4_human_ldl_predictions.csv", Path("ldl16")))

    for name in CODE:
        src = W / "code" / name
        if src.exists():
            shutil.copy2(src, OUT / "code" / name)
            copied.append(f"code/{name}")
        else:
            missing.append((f"code/{name}", src))

    shutil.rmtree(stage, ignore_errors=True)

    rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in ("MANIFEST.csv",) and not p.name.startswith(".stage"):
            rows.append({"path": str(p.relative_to(OUT)).replace("\\", "/"),
                         "bytes": p.stat().st_size, "sha256": sha256(p)})
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "MANIFEST.csv", index=False, encoding="utf-8-sig")

    print(f"\n{len(rows)} files, {man['bytes'].sum()/1e6:.1f} MB")
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for dest, src in missing:
            print(f"  {dest:52s} <- {src}")
    print()
    for p in sorted(OUT.iterdir()):
        if p.is_dir():
            n = sum(1 for q in p.rglob("*") if q.is_file())
            print(f"  {p.name + '/':18s} {n:3d} files")


if __name__ == "__main__":
    main()
