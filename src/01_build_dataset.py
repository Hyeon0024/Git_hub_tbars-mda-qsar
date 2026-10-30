"""lipid_peroxidation IC50 데이터셋 구축 (v2, 2026-10-01).

- IC50, 정확한 값(relation '='), 단위 nM 또는 ug/mL만 사용 (ug/mL는 분자량으로 nM 환산)
- [v2] 구조 표준화: 가장 큰 조각만 남김(염·짝이온 제거) + 전하 중화 -> parent SMILES
- [v2] 병합 기준을 ChEMBL ID -> 입체 무시 parent 구조로 변경
       (기술자·Morgan은 입체/염을 구분하지 못하므로, 같은 구조는 하나의 샘플로 병합)
- [v2] 논문 단위 이상치: 한 논문의 측정값 중 PAPER_BAD_FRAC 이상이 pIC50 범위 밖이면
       그 논문의 측정값 전체를 제외 (단위 오류로 논문 전체가 어긋난 경우 대응)
- [v2] 무기 원소(C,H,N,O,S,P,할로겐 외) 포함 화합물 inorganic_flag
- [v2] 원문 단위 확인이 필요한 논문 suspect_flag (기본은 학습에 포함, 02에서 --exclude-suspect로 제외)
- 병합 pIC50 = 정상 측정값의 중앙값. 이상치는 삭제하지 않고 플래그만 남김
- assay 세부 유형(조직/유도방식/측정법) 태그 부여
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "aodb_by_assay" / "lipid_peroxidation" / "lipid_peroxidation_data.csv"
OUT = HERE / "lipid_peroxidation_ic50.csv"
OUT_MEAS = HERE / "lipid_peroxidation_measurements.csv"  # 측정값 단위 상세 (검토용)

PIC50_LOW, PIC50_HIGH = 2.0, 9.5  # 이 범위 밖은 단위/입력 오류 의심
PAPER_BAD_FRAC = 0.5              # 논문 내 범위 밖 비율이 이 이상이면 논문 전체 제외
ORGANIC = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
# 원문 단위 확인 필요 (값 분포가 비정상, 단위 오류 의심) -> 확인 후 수정/삭제
SUSPECT_DOIS = {
    "10.1021/jm960772z",           # pIC50 8~9.2가 이 논문에만 집중, 계단식 값 (uM 오기 의심)
    "10.1016/j.bmcl.2012.05.061",  # 14개 전부 pIC50 < 2 (x1000 단위 오류 의심)
}


def tag_tissue(desc: str) -> str:
    d = desc.lower()
    if "microsom" in d:
        return "microsome"
    if "liposome" in d or "phospholipid" in d:
        return "liposome"
    if "brain" in d:
        return "brain"
    if "liver" in d or "homogenate" in d:
        return "liver"
    return "other"


def tag_inducer(desc: str) -> str:
    d = desc.lower()
    if any(k in d for k in ("aaph", "amvn", "abap", "azobis")):
        return "azo_initiator"
    if any(k in d for k in ("fe2", "fe(ii)", "feso4", "fecl", "ascorb", "iron")):
        return "fe_ascorbate"
    return "other"


def tag_method(desc: str) -> str:
    d = desc.lower()
    return "tbars_mda" if ("tbars" in d or "malondialdehyde" in d or "mda" in d.split()) else "other"


_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize(smi: str) -> tuple[str | None, str | None, bool, bool]:
    """-> (parent SMILES(입체 유지), 병합 키(입체 무시), 염/혼합물 여부, 무기 원소 포함 여부)"""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None, None, False, False
    multi = len(Chem.GetMolFrags(mol)) > 1
    parent = _uncharger.uncharge(_chooser.choose(mol))
    inorganic = any(a.GetSymbol() not in ORGANIC for a in parent.GetAtoms())
    return (Chem.MolToSmiles(parent), Chem.MolToSmiles(parent, isomericSmiles=False), multi, inorganic)


def main(source: Path = SRC, output: Path = OUT,
         measurements_output: Path = OUT_MEAS,
         method_filter: str | None = None) -> None:
    df = pd.read_csv(source)
    n_source = len(df)
    if method_filter is not None:
        if method_filter != "tbars_mda":
            raise ValueError(f"Unsupported assay method filter: {method_filter}")
        df = df[df["Assay Description"].astype(str).map(tag_method) == method_filter].copy()
    n0 = len(df)
    df = df[(df["value_type"] == "IC50") & (df["relation"] == "=")].copy()
    n_ic50 = len(df)

    mw = pd.to_numeric(df["Molecular Weight"], errors="coerce")
    nm = pd.Series(np.nan, index=df.index)
    nm[df["unit"] == "nM"] = df.loc[df["unit"] == "nM", "value"]
    ug = df["unit"] == "ug.mL-1"
    nm[ug] = df.loc[ug, "value"] / mw[ug] * 1e6  # ug/mL -> nM (DB 분자량 = 염 포함 분자량)
    df["ic50_nM"] = nm
    df = df.dropna(subset=["ic50_nM"])
    df = df[df["ic50_nM"] > 0].copy()
    df["pic50"] = -np.log10(df["ic50_nM"] * 1e-9)
    df["from_ug_ml"] = ug.reindex(df.index)
    df["DOI"] = df["DOI"].fillna("unknown")

    # --- 구조 표준화 ---
    std = df["SMILES"].map(standardize)
    df["parent_smiles"] = std.str[0]
    df["struct_key"] = std.str[1]
    df["is_salt_mixture"] = std.str[2]
    df["inorganic_flag"] = std.str[3]
    n_bad_smi = int(df["struct_key"].isna().sum())
    df = df.dropna(subset=["struct_key"]).copy()

    # --- 측정값 단위 이상치: 범위 밖 + 논문 단위 ---
    df["range_bad"] = ~df["pic50"].between(PIC50_LOW, PIC50_HIGH)
    bad_frac = df.groupby("DOI")["range_bad"].mean()
    bad_papers = set(bad_frac[bad_frac >= PAPER_BAD_FRAC].index)
    df["paper_bad"] = df["DOI"].isin(bad_papers)
    df["meas_bad"] = df["range_bad"] | df["paper_bad"]

    desc = df["Assay Description"].astype(str)
    df["tissue"] = desc.map(tag_tissue)
    df["inducer"] = desc.map(tag_inducer)
    df["method"] = desc.map(tag_method)

    def merge(g: pd.DataFrame) -> pd.Series:
        good = g[~g["meas_bad"]]
        use = good if len(good) else g            # 정상 측정값이 있으면 그것만 사용
        top_doi = use["DOI"].mode().iloc[0]
        row = use[use["DOI"] == top_doi].iloc[0]
        reasons = []
        if not len(good):
            if g["paper_bad"].any():
                reasons.append("paper")
            if g["range_bad"].any():
                reasons.append("range")
        return pd.Series({
            "ChEMBL ID": row["ChEMBL ID"],
            "chembl_ids": ";".join(sorted(g["ChEMBL ID"].astype(str).unique())),
            "smiles": row["parent_smiles"], "orig_smiles": row["SMILES"], "doi": top_doi,
            "pic50": use["pic50"].median(), "n_meas": len(use), "n_meas_dropped": len(g) - len(use),
            "pic50_std": use["pic50"].std() if len(use) > 1 else np.nan,
            "tissue": row["tissue"], "inducer": row["inducer"], "method": row["method"],
            "from_ug_ml": bool(use["from_ug_ml"].any()),
            "is_salt_mixture": bool(g["is_salt_mixture"].any()),
            "inorganic_flag": bool(row["inorganic_flag"]),
            "suspect_flag": top_doi in SUSPECT_DOIS,
            "outlier_flag": not len(good),
            "outlier_reason": "+".join(reasons),
        })

    out = df.groupby("struct_key").apply(merge, include_groups=False).reset_index()
    out = out[["ChEMBL ID", "smiles", "doi", "pic50", "n_meas", "pic50_std", "tissue", "inducer", "method",
               "from_ug_ml", "outlier_flag", "outlier_reason", "suspect_flag", "inorganic_flag",
               "is_salt_mixture", "n_meas_dropped", "chembl_ids", "orig_smiles", "struct_key"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    measurements_output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    df.drop(columns=["Assay Description"], errors="ignore").to_csv(
        measurements_output, index=False, encoding="utf-8-sig")

    n_cid = df["ChEMBL ID"].nunique()
    print(f"lipid_peroxidation 원자료 전체 행: {n_source}")
    if method_filter is not None:
        print(f"시험법 태그 {method_filter} 우선 선별 후: {n0}행")
    print(f"IC50 & '=' 행: {n_ic50}  -> nM 환산 가능: {len(df) + n_bad_smi}  (ug/mL 환산 {int(df['from_ug_ml'].sum())}행)")
    print(f"SMILES 파싱 실패: {n_bad_smi}행 / 염·혼합물 SMILES: {int(df['is_salt_mixture'].sum())}행 (가장 큰 조각으로 표준화)")
    print(f"ChEMBL ID {n_cid}개 -> 구조 병합 후 {len(out)}개 (구조 중복으로 {n_cid - len(out)}개 합쳐짐, "
          f"여러 ID가 합쳐진 구조 {int(out.chembl_ids.str.contains(';').sum())}개)")
    print(f"중복 측정 화합물: {int((out.n_meas > 1).sum())}개")
    print(f"범위 밖 측정값 [{PIC50_LOW}, {PIC50_HIGH}]: {int(df['range_bad'].sum())}행")
    print(f"논문 단위 제외 (범위 밖 비율 >= {PAPER_BAD_FRAC}): {len(bad_papers)}편, {int(df['paper_bad'].sum())}행")
    for d in sorted(bad_papers):
        print(f"   - {d}  ({int((df.DOI == d).sum())}행, 범위 밖 {bad_frac[d]:.0%})")
    print(f"outlier_flag (학습 제외): {int(out.outlier_flag.sum())}개  "
          f"{out.loc[out.outlier_flag, 'outlier_reason'].value_counts().to_dict()}")
    print(f"측정값 일부만 제외된 화합물: {int(((out.n_meas_dropped > 0) & ~out.outlier_flag).sum())}개")
    print(f"suspect_flag (원문 확인 필요): {int(out.suspect_flag.sum())}개 / inorganic_flag: {int(out.inorganic_flag.sum())}개")
    print(f"논문(DOI) 수: {out.doi.nunique()}")
    print("\n태그 분포:")
    for c in ("tissue", "inducer", "method"):
        print(out[c].value_counts().to_string(), "\n")
    print(f"저장: {output}\n      {measurements_output}")


if __name__ == "__main__":
    # CLI added for the public repository; the curation logic is unchanged.
    import argparse

    _p = argparse.ArgumentParser(description=__doc__)
    _p.add_argument("--source", type=Path, default=SRC,
                    help="retrieved records (data/raw_records_4470.csv)")
    _p.add_argument("--output", type=Path, default=OUT,
                    help="curated structures (data/curated_structures_378.csv)")
    _p.add_argument("--measurements-output", type=Path, default=OUT_MEAS,
                    help="curated measurements (data/curated_measurements_443.csv)")
    _a = _p.parse_args()
    main(source=_a.source, output=_a.output,
         measurements_output=_a.measurements_output)
