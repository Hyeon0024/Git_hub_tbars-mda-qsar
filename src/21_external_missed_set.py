"""External validation on the records the assay-description text rule missed.

The screening rule in 01_build_dataset.py tags a record as TBARS/MDA when its
description contains 'tbars', 'malondialdehyde', or a standalone 'mda' token.
Descriptions that spell out 'thiobarbituric acid' without the acronym are not
tagged. Those records satisfy the stated inclusion criterion of the study but
were absent from the curated set, so they form an external test set whose
membership is fixed by an objective rule rather than by inspection: every
missed record with an exact IC50, from publications not used in training.

Builds that set with the same curation steps as the training data, applies the
descriptor model without refitting, and reports the error and its direction
against the internal holdout and the human-LDL panel.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
NJOBS = int(os.environ.get("AUDIT_NJOBS", "1"))

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "st", str(HERE / "15_rdkit_descriptor_study.py"))
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

PIC50_LOW, PIC50_HIGH = 2.0, 9.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True,
                   help="raw AODB lipid-peroxidation export")
    p.add_argument("--data-file", type=Path, required=True,
                   help="curated 378-structure training table")
    p.add_argument("--study-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    builder = runpy.run_path(str(HERE / "01_build_dataset.py"))
    rep = json.loads(args.study_report.read_text(encoding="utf-8"))
    params = rep["selected_model"]["params"]
    cutoff = rep["applicability_domain"]["cutoff"]

    # ---- curated training data, refit the published model -----------------
    d = pd.read_csv(args.data_file)
    d = d[d["method"] == "tbars_mda"]
    d = d[d["pic50"].between(PIC50_LOW, PIC50_HIGH)]
    d = d[~d["outlier_flag"].astype(bool)].reset_index(drop=True)
    y = d["pic50"].to_numpy(dtype=float)
    X = st.descriptor_block(d["smiles"])
    idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=0.2, random_state=args.seed,
                              shuffle=True)
    tr, te = np.sort(tr), np.sort(te)
    pipe = st.build_estimator("gradient_boosting", list(X.columns)).set_params(**params)
    pipe.fit(X.iloc[tr], y[tr])
    train_mean = float(y[tr].mean())
    print(f"model refit on {len(tr)} training structures; training mean pIC50 "
          f"{train_mean:.4f}", flush=True)

    # ---- the missed records ----------------------------------------------
    raw = pd.read_csv(args.source, low_memory=False)
    desc = raw["Assay Description"].astype(str)
    tagged = desc.map(builder["tag_method"]).eq("tbars_mda")
    spells_out = desc.str.lower().str.contains("thiobarbituric")
    missed = spells_out & ~tagged
    print(f"records naming thiobarbituric acid: {int(spells_out.sum())}; "
          f"tagged by the text rule: {int((spells_out & tagged).sum())}; "
          f"missed: {int(missed.sum())}", flush=True)

    ext = raw[missed & raw["value_type"].eq("IC50") & raw["relation"].eq("=")].copy()
    mw = pd.to_numeric(ext["Molecular Weight"], errors="coerce")
    nm = pd.Series(np.nan, index=ext.index, dtype=float)
    is_nm = ext["unit"].eq("nM")
    is_ug = ext["unit"].eq("ug.mL-1")
    nm[is_nm] = ext.loc[is_nm, "value"].astype(float)
    nm[is_ug] = ext.loc[is_ug, "value"].astype(float) / mw[is_ug] * 1e6
    ext["ic50_nM"] = nm
    ext = ext[ext["ic50_nM"] > 0].copy()
    ext["pic50_meas"] = 9 - np.log10(ext["ic50_nM"])
    ext["from_ug_ml"] = is_ug.reindex(ext.index).fillna(False)

    std = ext["SMILES"].map(builder["standardize"])
    ext["parent_smiles"] = std.str[0]
    ext["struct_key"] = std.str[1]
    ext = ext.dropna(subset=["struct_key"])
    ext["tissue"] = ext["Assay Description"].astype(str).map(builder["tag_tissue"])
    ext["inducer"] = ext["Assay Description"].astype(str).map(builder["tag_inducer"])

    n_before = ext["struct_key"].nunique()
    in_range = ext["pic50_meas"].between(PIC50_LOW, PIC50_HIGH)
    print(f"exact IC50 rows {len(ext)} from {ext['DOI'].nunique()} publications; "
          f"{int((~in_range).sum())} outside the pIC50 window were dropped",
          flush=True)
    ext = ext[in_range]

    # aggregate to structure level exactly as the training data was
    rows = []
    for key, g in ext.groupby("struct_key"):
        rows.append({
            "struct_key": key,
            "parent_smiles": g["parent_smiles"].iloc[0],
            "chembl_ids": ";".join(sorted(set(g["ChEMBL ID"].astype(str)))),
            "doi": g["DOI"].mode().iloc[0],
            "n_dois": g["DOI"].nunique(),
            "n_meas": len(g),
            "observed_pic50": float(g["pic50_meas"].median()),
            "pic50_spread": float(g["pic50_meas"].max() - g["pic50_meas"].min()),
            "tissue": g["tissue"].mode().iloc[0],
            "inducer": g["inducer"].mode().iloc[0],
            "assay_description": g["Assay Description"].iloc[0],
            "any_mass_conversion": bool(g["from_ug_ml"].any()),
        })
    e = pd.DataFrame(rows)
    train_keys = set(d["struct_key"])
    e["in_curated_dataset"] = e["struct_key"].isin(train_keys)
    print(f"structures after aggregation: {len(e)}; already in the curated set: "
          f"{int(e['in_curated_dataset'].sum())}", flush=True)
    e = e[~e["in_curated_dataset"]].reset_index(drop=True)

    train_dois = set(d["doi"].astype(str))
    e["doi_in_training"] = e["doi"].astype(str).isin(train_dois)
    print(f"external structures: {len(e)} from {e['doi'].nunique()} publications; "
          f"publications overlapping training: {int(e['doi_in_training'].sum())}",
          flush=True)

    # ---- predict ----------------------------------------------------------
    Xe = st.descriptor_block(e["parent_smiles"]).reindex(columns=X.columns)
    # Ipc grows factorially with graph size and can exceed the float32 range the
    # boosted trees cast to. No training structure does this (checked: 0 values),
    # so blanking those cells here leaves the fitted model untouched and lets the
    # fold-fitted median imputer fill them.
    limit = float(np.finfo(np.float32).max)
    overflow = Xe.abs() > limit
    if int(overflow.to_numpy().sum()):
        cols = sorted(Xe.columns[overflow.any(axis=0)])
        print(f"  blanked {int(overflow.to_numpy().sum())} descriptor values beyond "
              f"the float32 range in {cols}", flush=True)
        Xe = Xe.mask(overflow)
    e["predicted_pic50"] = pipe.predict(Xe)
    e["signed_error"] = e["predicted_pic50"] - e["observed_pic50"]
    e["absolute_error"] = e["signed_error"].abs()
    e["fold_error_ic50"] = 10 ** e["absolute_error"]

    prep = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("variance", VarianceThreshold(0.0)),
                     ("scale", StandardScaler())])
    prep.fit(X.iloc[tr])
    Ztr, Ze = prep.transform(X.iloc[tr]), prep.transform(Xe)
    e["nearest_training_distance"] = [
        float(np.sqrt(((Ztr - z) ** 2).sum(axis=1)).min()) for z in Ze]
    e["within_domain"] = e["nearest_training_distance"] <= cutoff

    e.to_csv(out / "external_missed_predictions.csv", index=False,
             encoding="utf-8-sig")

    def block(frame, label):
        if frame.empty:
            return {"label": label, "n": 0}
        o, pr = frame["observed_pic50"].to_numpy(), frame["predicted_pic50"].to_numpy()
        return {"label": label, "n": int(len(frame)),
                "r2": float(r2_score(o, pr)) if len(frame) > 1 else None,
                "rmse": float(mean_squared_error(o, pr) ** 0.5),
                "mae": float(mean_absolute_error(o, pr)),
                "mean_signed_error": float((pr - o).mean()),
                "median_signed_error": float(np.median(pr - o)),
                "n_underpredicted": int((pr < o).sum()),
                "observed_mean": float(o.mean()),
                "offset_from_training_mean": float(o.mean() - train_mean)}

    summary = {
        "text_rule": {
            "n_naming_thiobarbituric": int(spells_out.sum()),
            "n_tagged_by_rule": int((spells_out & tagged).sum()),
            "n_missed": int(missed.sum()),
        },
        "external_set": {
            "n_structures": int(len(e)),
            "n_publications": int(e["doi"].nunique()),
            "publications": sorted(e["doi"].astype(str).unique().tolist()),
            "tissue_counts": e["tissue"].value_counts().to_dict(),
            "inducer_counts": e["inducer"].value_counts().to_dict(),
            "n_within_domain": int(e["within_domain"].sum()),
        },
        "performance": {
            "all": block(e, "all external"),
            "within_domain": block(e[e["within_domain"]], "within domain"),
            "outside_domain": block(e[~e["within_domain"]], "outside domain"),
            "brain_only": block(e[e["tissue"] == "brain"], "brain tissue only"),
        },
        "training_mean_pic50": train_mean,
    }
    (out / "external_missed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== external set ===")
    show = ["struct_key", "doi", "tissue", "observed_pic50", "predicted_pic50",
            "signed_error", "fold_error_ic50", "within_domain"]
    pd.set_option("display.width", 200)
    print(e[show].assign(struct_key=e["struct_key"].str[:30]).round(3)
          .sort_values("signed_error").to_string(index=False))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
