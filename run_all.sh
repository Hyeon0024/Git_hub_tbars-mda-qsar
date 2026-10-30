#!/usr/bin/env bash
# Reproduce every reported number from the curated data.
#
# Results are written to rerun/ so that the published results/ tree is left
# untouched and the two can be compared. Run `./run_all.sh --check` to compare
# the key metrics afterwards.
#
# The full run takes roughly 30-60 minutes on one core; the model-selection
# step dominates. Set AUDIT_NJOBS to use more cores for the tree ensembles
# (it does not change the results).
set -euo pipefail

PY=${PY:-python}
OUT=${OUT:-rerun}
SEED=${SEED:-42}
export AUDIT_NJOBS=${AUDIT_NJOBS:-1}

if [[ "${1:-}" == "--check" ]]; then
    $PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
new = Path(sys.argv[1]) / "rdkit_study_report.json"
old = Path("results/reports/rdkit_study_report.json")
if not new.exists():
    sys.exit(f"no rerun report at {new} — run ./run_all.sh first")
a, b = json.loads(new.read_text()), json.loads(old.read_text())
checks = [
    ("selected family",        a["selected_model"]["family"],        b["selected_model"]["family"]),
    ("training CV R2",         a["selected_model"]["cv_r2_mean"],    b["selected_model"]["cv_r2_mean"]),
    ("pooled OOF R2",          a["random_design"]["pooled_oof"]["r2"], b["random_design"]["pooled_oof"]["r2"]),
    ("holdout R2",             a["random_design"]["holdout"]["r2"],  b["random_design"]["holdout"]["r2"]),
    ("holdout RMSE",           a["random_design"]["holdout"]["rmse"], b["random_design"]["holdout"]["rmse"]),
    ("scaffold-grouped R2",    a["scaffold_design"]["pooled"]["r2"], b["scaffold_design"]["pooled"]["r2"]),
    ("1-NN holdout R2",        a["random_design"]["nn_reference"]["r2"], b["random_design"]["nn_reference"]["r2"]),
    ("AD cutoff",              a["applicability_domain"]["cutoff"],  b["applicability_domain"]["cutoff"]),
]
bad = 0
print(f"{'quantity':24s} {'rerun':>12s} {'published':>12s}  status")
print("-" * 66)
for label, got, exp in checks:
    if isinstance(got, str):
        ok = got == exp
        print(f"{label:24s} {got:>12s} {exp:>12s}  {'ok' if ok else 'DIFFERS'}")
    else:
        ok = abs(got - exp) < 5e-4
        print(f"{label:24s} {got:>12.4f} {exp:>12.4f}  {'ok' if ok else 'DIFFERS'}")
    bad += not ok
print()
print("all key metrics reproduce" if not bad else
      f"{bad} metric(s) differ — check the package versions in requirements.txt")
sys.exit(1 if bad else 0)
EOF
    exit $?
fi

mkdir -p "$OUT"

echo "=== [1/6] curation: raw records -> 443 measurements -> 378 structures"
$PY src/01_build_dataset.py \
    --source data/raw_records_4470.csv \
    --output "$OUT/curated_structures_378.csv" \
    --measurements-output "$OUT/curated_measurements_443.csv"

echo "=== [2/6] assay-protocol audit (Tables S2-S4)"
$PY src/11_protocol_audit.py \
    --source data/raw_records_4470.csv --output-dir "$OUT/audit"

echo "=== [3/6] the study: model selection, both designs, applicability domain, SHAP"
$PY src/15_rdkit_descriptor_study.py \
    --data-file data/curated_structures_378.csv \
    --output-dir "$OUT" --seed "$SEED" --search-iterations 40

echo "=== [4/6] Figures 1-4"
$PY src/16_rdkit_figures.py --source "$OUT" --output-dir "$OUT/figures"

echo "=== [5/6] Table S1, Figures S1-S3, and the serialized model"
$PY src/26_build_deposit.py \
    --data-file data/curated_structures_378.csv \
    --study-dir "$OUT" --output-dir "$OUT/extras" --seed "$SEED"

echo "=== [6/6] external test set (Table S5) and the human-LDL predictions (Table 4)"
$PY src/21_external_missed_set.py \
    --source data/raw_records_4470.csv \
    --data-file data/curated_structures_378.csv \
    --study-report "$OUT/rdkit_study_report.json" \
    --output-dir "$OUT/external" --seed "$SEED"
$PY src/20_ldl_candidates.py \
    --data-file data/curated_structures_378.csv \
    --candidates data/ldl_literature_candidates.csv \
    --study-report "$OUT/rdkit_study_report.json" \
    --output-dir "$OUT/ldl" --seed "$SEED"

echo
echo "=== comparing against the published results"
"$0" --check
