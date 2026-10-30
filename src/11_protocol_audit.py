"""Phase 11: protocol audit table for the 443 selected IC50 measurements
(pre-submission review, priority 1 / predicted Major comment 1).

Links every retained measurement to its source publication and to the assay
conditions as stated in the AODB assay description: lipid substrate / sample
matrix, species and strain, oxidation initiator, reaction time, analytical
readout, and the originally reported value and unit.

Conditions that the source description does not state are recorded as
"not reported" rather than inferred. The readout classification separates
descriptions that name TBARS from those that name malondialdehyde, and
flags that neither wording establishes whether MDA was chromatographically
separated before quantification.
"""
from __future__ import annotations

import argparse
import json
import re
import runpy
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# condition parsers - keyword rules applied to the assay description
# ---------------------------------------------------------------------------

# ordered: first match wins, most specific first
MATRIX_RULES = [
    ("human LDL", r"human ldl|ldl[- ]oxidation|\bldl\b"),
    ("liposome / phospholipid vesicle", r"liposome|phosphatidylcholine|phospholipid"),
    ("liver microsome", r"(liver|hepatic)\s+microsom"),
    ("brain microsome", r"brain\s+microsom"),
    ("microsome (organ not stated)", r"microsom"),
    ("brain homogenate", r"brain"),
    ("liver homogenate", r"liver|hepatic"),
    ("cultured cells", r"\bcells?\b|thp1|thp-1|macrophage"),
    ("erythrocyte / membrane", r"erythrocyte|red blood cell|\brbc\b"),
    ("mitochondria", r"mitochondri"),
    ("tissue homogenate (organ not stated)", r"homogenate"),
]

SPECIES_RULES = [
    ("rat (Fischer-344)", r"fischer[- ]?344"),
    ("rat (Wistar)", r"wistar"),
    ("rat (Sprague-Dawley)", r"sprague"),
    ("rat (strain not stated)", r"\brat\b|\brats\b"),
    ("mouse", r"\bmouse\b|\bmice\b|murine"),
    ("human", r"\bhuman\b"),
    ("bovine", r"\bbovine\b|\bcattle\b"),
    ("pig / porcine", r"porcine|\bpig\b"),
]

INITIATOR_RULES = [
    ("Fe(II)/ascorbate", r"ferrous salt/ascorb|ferrous.{0,20}ascorb|iron[- ]ascorb|fe\(ii\).{0,20}ascorb|fe2\+.{0,20}ascorb"),
    ("ascorbyl radical", r"ascorbyl radical"),
    ("ascorbate (iron not stated)", r"ascorb"),
    ("Cu(II)", r"copper|cuso4|cu2\+|cu\(ii\)"),
    ("tert-butyl hydroperoxide", r"tert-butylhydroperoxide|tert-butyl hydroperoxide|t-buooh|tbhp"),
    ("cumene hydroperoxide", r"cumene"),
    ("azo initiator (AAPH/AMVN/ABAP)", r"aaph|amvn|abap|azobis"),
    ("hydroxyl radical", r"hydroxyl radical"),
    ("peroxyl radical", r"peroxyl radical"),
    ("Fe(II) / iron (no reductant stated)", r"\bfe2\+|\bfecl|\bfeso4|\biron\b|\bferrous\b|\bfe\(ii\)"),
    ("NADPH-dependent (enzymatic)", r"nadph"),
    ("ADP-Fe", r"adp[- ]fe"),
    ("UV / photo-oxidation", r"\buv\b|photo[- ]?oxidat"),
    ("spontaneous / endogenous", r"spontaneous|endogenous|autoxidat"),
]

READOUT_RULES = [
    ("TBARS, spectrophotometric (stated)",
     r"tbars.{0,40}spectrophotometric|spectrophotometric.{0,40}tbars"),
    ("TBARS (method not further specified)", r"tbars"),
    ("MDA formation (separation not stated)", r"malondialdehyde|\bmda\b"),
]

TIME_RE = re.compile(
    r"(?:after|for|within)\s+(\d+(?:\.\d+)?)\s*"
    r"(min|mins|minute|minutes|hr|hrs|hour|hours|h|day|days|sec|secs|seconds)\b",
    re.I)


def _first_match(rules, text: str, default: str) -> str:
    for label, pattern in rules:
        if re.search(pattern, text, re.I):
            return label
    return default


def parse_time(text: str) -> tuple[str, float | None]:
    m = TIME_RE.search(text)
    if not m:
        return "not reported", None
    value, unit = float(m.group(1)), m.group(2).lower()
    minutes = {"sec": 1 / 60, "secs": 1 / 60, "seconds": 1 / 60,
               "min": 1, "mins": 1, "minute": 1, "minutes": 1,
               "hr": 60, "hrs": 60, "hour": 60, "hours": 60, "h": 60,
               "day": 1440, "days": 1440}[unit]
    total = value * minutes
    shown = f"{value:g} {unit}"
    return shown, total


def classify(description: str) -> dict:
    text = str(description)
    readout = _first_match(READOUT_RULES, text, "not reported")
    names_tbars = bool(re.search(r"tbars", text, re.I))
    names_mda = bool(re.search(r"malondialdehyde|\bmda\b", text, re.I))
    time_shown, time_minutes = parse_time(text)
    return {
        "matrix_substrate": _first_match(MATRIX_RULES, text, "not reported"),
        "species_strain": _first_match(SPECIES_RULES, text, "not reported"),
        "oxidation_initiator": _first_match(INITIATOR_RULES, text, "not reported"),
        "reaction_time": time_shown,
        "reaction_time_minutes": time_minutes,
        "analytical_readout": readout,
        "names_tbars": names_tbars,
        "names_mda": names_mda,
        "readout_wording": ("TBARS and MDA both named" if names_tbars and names_mda
                            else "TBARS named" if names_tbars
                            else "MDA named" if names_mda else "neither named"),
        "mda_separation_stated": False,  # no description states chromatographic separation
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    builder = runpy.run_path(str(HERE / "01_build_dataset.py"))
    raw = pd.read_csv(args.source, low_memory=False)
    tagged = raw["Assay Description"].astype(str).map(builder["tag_method"]).eq("tbars_mda")
    mask = tagged & raw["value_type"].eq("IC50") & raw["relation"].eq("=")
    sel = raw.loc[mask].copy()
    sel.insert(0, "source_row_index", sel.index)
    print(f"selected measurements: {len(sel)}  unique descriptions: "
          f"{sel['Assay Description'].nunique()}  DOIs: {sel['DOI'].nunique()}")

    parsed = pd.DataFrame([classify(t) for t in sel["Assay Description"]],
                          index=sel.index)
    audit = pd.concat([sel, parsed], axis=1)

    # recompute the molar endpoint exactly as the build script does, so the
    # audit row carries the number that entered the model
    mw = pd.to_numeric(audit["Molecular Weight"], errors="coerce")
    nm = pd.Series(pd.NA, index=audit.index, dtype="Float64")
    is_nm = audit["unit"].eq("nM")
    is_ug = audit["unit"].eq("ug.mL-1")
    nm[is_nm] = audit.loc[is_nm, "value"].astype(float)
    nm[is_ug] = audit.loc[is_ug, "value"].astype(float) / mw[is_ug] * 1e6
    audit["reported_value"] = audit["value"]
    audit["reported_unit"] = audit["unit"]
    audit["ic50_nM"] = nm
    audit["pic50"] = 9 - pd.to_numeric(audit["ic50_nM"], errors="coerce").map(
        lambda v: pd.NA if pd.isna(v) else __import__("math").log10(v))
    audit["converted_from_mass_concentration"] = is_ug
    audit["db_molecular_weight"] = mw
    audit["tissue_tag_in_manuscript"] = audit["Assay Description"].astype(str).map(
        builder["tag_tissue"])
    audit["inducer_tag_in_manuscript"] = audit["Assay Description"].astype(str).map(
        builder["tag_inducer"])

    std = audit["SMILES"].map(builder["standardize"])
    audit["parent_smiles"] = std.str[0]
    audit["struct_key"] = std.str[1]
    audit["salt_or_mixture_smiles"] = std.str[2]

    cols = ["source_row_index", "AOX ID", "ChEMBL ID", "PMID", "DOI",
            "Assay Description", "matrix_substrate", "species_strain",
            "oxidation_initiator", "reaction_time", "reaction_time_minutes",
            "analytical_readout", "readout_wording", "names_tbars", "names_mda",
            "mda_separation_stated", "reported_value", "reported_unit",
            "converted_from_mass_concentration", "db_molecular_weight",
            "ic50_nM", "pic50", "salt_or_mixture_smiles", "SMILES",
            "parent_smiles", "struct_key",
            "tissue_tag_in_manuscript", "inducer_tag_in_manuscript"]
    audit[cols].to_csv(out / "TableS3_protocol_audit_443.csv",
                       index=False, encoding="utf-8-sig")

    # ---- per-description summary (47 rows) ---------------------------
    desc = (audit.groupby(["Assay Description", "matrix_substrate",
                           "species_strain", "oxidation_initiator",
                           "reaction_time", "analytical_readout",
                           "readout_wording"], dropna=False)
            .agg(n_measurements=("source_row_index", "size"),
                 n_structures=("struct_key", "nunique"),
                 n_dois=("DOI", "nunique"),
                 dois=("DOI", lambda s: ";".join(sorted(set(s.dropna().astype(str))))),
                 pic50_min=("pic50", "min"), pic50_max=("pic50", "max"))
            .reset_index()
            .sort_values("n_measurements", ascending=False))
    desc.to_csv(out / "TableS3b_description_summary.csv",
                index=False, encoding="utf-8-sig")

    # ---- condition coverage ------------------------------------------
    coverage = {}
    for column in ("matrix_substrate", "species_strain", "oxidation_initiator",
                   "reaction_time", "analytical_readout", "readout_wording"):
        counts = audit[column].value_counts(dropna=False)
        coverage[column] = {str(k): int(v) for k, v in counts.items()}
    coverage["not_reported_counts"] = {
        column: int((audit[column] == "not reported").sum())
        for column in ("matrix_substrate", "species_strain",
                       "oxidation_initiator", "reaction_time",
                       "analytical_readout")}

    # ---- multi-measurement structures with their conditions ----------
    grp = audit.groupby("struct_key")
    multi_keys = [k for k, g in grp if len(g) > 1]
    rows = []
    for key in multi_keys:
        g = audit[audit["struct_key"] == key]
        rows.append({
            "struct_key": key,
            "parent_smiles": g["parent_smiles"].iloc[0],
            "chembl_ids": ";".join(sorted(set(g["ChEMBL ID"].astype(str)))),
            "n_measurements": len(g),
            "n_dois": g["DOI"].nunique(),
            "dois": ";".join(sorted(set(g["DOI"].dropna().astype(str)))),
            "n_distinct_matrices": g["matrix_substrate"].nunique(),
            "matrices": ";".join(sorted(set(g["matrix_substrate"]))),
            "n_distinct_initiators": g["oxidation_initiator"].nunique(),
            "initiators": ";".join(sorted(set(g["oxidation_initiator"]))),
            "n_distinct_times": g["reaction_time"].nunique(),
            "times": ";".join(sorted(set(g["reaction_time"].astype(str)))),
            "n_distinct_readouts": g["analytical_readout"].nunique(),
            "readouts": ";".join(sorted(set(g["analytical_readout"]))),
            "pic50_values": ";".join(f"{v:.3f}" for v in sorted(g["pic50"].dropna())),
            "pic50_min": float(g["pic50"].min()),
            "pic50_max": float(g["pic50"].max()),
            "pic50_span": float(g["pic50"].max() - g["pic50"].min()),
            "pic50_median_used": float(g["pic50"].median()),
            "fold_range_ic50": float(10 ** (g["pic50"].max() - g["pic50"].min())),
            "any_mass_conversion": bool(g["converted_from_mass_concentration"].any()),
        })
    multi = pd.DataFrame(rows).sort_values("pic50_span", ascending=False)
    multi.to_csv(out / "TableS3c_multi_measurement_conditions.csv",
                 index=False, encoding="utf-8-sig")

    # replicate vs cross-protocol classification
    def kind(r):
        same_protocol = (r["n_distinct_matrices"] == 1
                         and r["n_distinct_initiators"] == 1
                         and r["n_distinct_times"] == 1
                         and r["n_distinct_readouts"] == 1)
        if same_protocol and r["n_dois"] == 1:
            return "same description, same publication"
        if same_protocol:
            return "same description, different publications"
        return "different assay conditions"
    multi["pooling_kind"] = multi.apply(kind, axis=1)

    summary = {
        "n_measurements": int(len(audit)),
        "n_structures": int(audit["struct_key"].nunique()),
        "n_unique_descriptions": int(audit["Assay Description"].nunique()),
        "n_dois": int(audit["DOI"].nunique()),
        "n_mass_concentration_conversions": int(is_ug.sum()),
        "n_salt_or_mixture_smiles": int(audit["salt_or_mixture_smiles"].sum()),
        "coverage": coverage,
        "multi_measurement": {
            "n_structures": int(len(multi)),
            "by_pooling_kind": {k: int(v) for k, v in
                                multi["pooling_kind"].value_counts().items()},
            "pic50_span_median": float(multi["pic50_span"].median()),
            "pic50_span_max": float(multi["pic50_span"].max()),
            "n_span_above_1_log": int((multi["pic50_span"] > 1).sum()),
            "fold_range_max": float(multi["fold_range_ic50"].max()),
        },
    }
    (out / "protocol_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    multi.to_csv(out / "TableS3c_multi_measurement_conditions.csv",
                 index=False, encoding="utf-8-sig")

    print("\n=== condition coverage ===")
    for column, counts in coverage.items():
        if column == "not_reported_counts":
            continue
        print(f"\n{column}:")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"   {v:4d}  {k}")
    print("\n=== not reported ===")
    print(json.dumps(coverage["not_reported_counts"], indent=2))
    print("\n=== multi-measurement structures ===")
    print(json.dumps(summary["multi_measurement"], indent=2, ensure_ascii=False))
    print("\nwidest-spread pooled structures:")
    show = ["struct_key", "n_measurements", "pooling_kind", "pic50_span",
            "fold_range_ic50", "matrices", "initiators", "times"]
    print(multi[show].head(12).to_string(index=False, max_colwidth=38))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
