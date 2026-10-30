"""Figures for the RDKit-descriptor version of the manuscript.

Figure 1  observed vs predicted pIC50 (pooled out-of-fold | internal holdout)
Figure 2  applicability domain: nearest-training distance vs absolute error
Figure 3  global SHAP importance (15 highest-ranked descriptors)
Figure 4  SHAP beeswarm for the same descriptors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
})
INK = "#1f2d3d"
ACCENT = "#2b6cb0"
WARN = "#c05621"


def parity(ax, obs, pred, title, label):
    lo = min(obs.min(), pred.min()) - 0.25
    hi = max(obs.max(), pred.max()) + 0.25
    ax.plot([lo, hi], [lo, hi], color="#9aa5b1", lw=0.9, ls="--", zorder=1)
    ax.scatter(obs, pred, s=16, facecolor=ACCENT, edgecolor="white",
               linewidth=0.4, alpha=0.85, zorder=2)
    r2 = 1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
    rmse = float(np.sqrt(((obs - pred) ** 2).mean()))
    mae = float(np.abs(obs - pred).mean())
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed pIC$_{50}$")
    ax.set_ylabel("Predicted pIC$_{50}$")
    ax.set_title(title, loc="left", color=INK)
    ax.text(0.04, 0.96,
            f"$n$ = {len(obs)}\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, va="top", ha="left", color=INK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cbd2d9", linewidth=0.6))
    ax.text(-0.17, 1.04, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure1(out: Path, src: Path):
    oof = pd.read_csv(src / "training_oof_predictions.csv")
    hold = pd.read_csv(src / "holdout_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6))
    parity(axes[0], oof["observed_pic50"].to_numpy(),
           oof["oof_prediction"].to_numpy(),
           "Training set, pooled five-fold out-of-fold", "A")
    parity(axes[1], hold["observed_pic50"].to_numpy(),
           hold["predicted_pic50"].to_numpy(),
           "Fixed internal holdout", "B")
    fig.tight_layout()
    fig.savefig(out / "Figure1_observed_vs_predicted.png")
    plt.close(fig)


def figure2(out: Path, src: Path):
    ad = pd.read_csv(src / "applicability_domain_holdout.csv")
    rep = json.loads((src / "applicability_domain_report.json").read_text())
    cutoff = rep["cutoff"]
    inside = ad["within_domain"].astype(bool)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.axvline(cutoff, color="#9aa5b1", lw=0.9, ls="--", zorder=1)
    ax.scatter(ad.loc[inside, "nearest_training_distance"],
               ad.loc[inside, "absolute_error"], s=18, facecolor=ACCENT,
               edgecolor="white", linewidth=0.4, label=f"inside ($n$ = {int(inside.sum())})",
               zorder=2)
    ax.scatter(ad.loc[~inside, "nearest_training_distance"],
               ad.loc[~inside, "absolute_error"], s=30, facecolor=WARN,
               edgecolor="white", linewidth=0.5, marker="D",
               label=f"outside ($n$ = {int((~inside).sum())})", zorder=3)
    ax.annotate(f"cutoff = {cutoff:.2f}", xy=(cutoff, ax.get_ylim()[1]),
                xytext=(4, -10), textcoords="offset points", fontsize=8,
                color="#52606d", va="top")
    ax.set_xlabel("Nearest training distance (standardized descriptor space)")
    ax.set_ylabel("Absolute error (pIC$_{50}$)")
    ax.legend(frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "Figure2_applicability_domain.png")
    plt.close(fig)


def figure3(out: Path, src: Path, top: int = 15):
    imp = pd.read_csv(src / "shap_descriptor_importance.csv").head(top)
    imp = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.barh(imp["descriptor"], imp["mean_abs_shap"], color=ACCENT,
            edgecolor="white", linewidth=0.5, height=0.72)
    for y, v in zip(range(len(imp)), imp["mean_abs_shap"]):
        ax.text(v + imp["mean_abs_shap"].max() * 0.015, y, f"{v:.3f}",
                va="center", fontsize=7, color="#52606d")
    ax.set_xlabel("Mean |SHAP value| (pIC$_{50}$ units)")
    ax.set_xlim(0, imp["mean_abs_shap"].max() * 1.16)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "Figure3_shap_importance.png")
    plt.close(fig)


def figure4(out: Path, src: Path, top: int = 15):
    sv = pd.read_csv(src / "shap_values_holdout.csv")
    imp = pd.read_csv(src / "shap_descriptor_importance.csv").head(top)
    desc = pd.read_csv(src / "descriptors_378.csv")
    manifest = pd.read_csv(src / "split_manifest.csv")
    te = manifest.index[manifest["partition"] == "test"].to_numpy()
    cols = [c for c in imp["descriptor"] if c in sv.columns][::-1]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    rng = np.random.default_rng(42)
    for i, col in enumerate(cols):
        vals = sv[col].to_numpy()
        raw = desc[col].to_numpy()[te].astype(float)
        finite = np.isfinite(raw)
        norm = np.full_like(raw, 0.5, dtype=float)
        if finite.sum() > 1:
            lo, hi = float(np.min(raw[finite])), float(np.max(raw[finite]))
            if hi > lo:
                norm[finite] = (raw[finite] - lo) / (hi - lo)
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        sc = ax.scatter(vals, np.full(len(vals), i) + jitter, c=norm,
                        cmap="coolwarm", s=11, linewidth=0.2,
                        edgecolor="white", vmin=0, vmax=1, alpha=0.9)
    ax.axvline(0, color="#9aa5b1", lw=0.8, zorder=0)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xlabel("SHAP contribution to predicted pIC$_{50}$")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label("Descriptor value (low → high)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    fig.tight_layout()
    fig.savefig(out / "Figure4_shap_beeswarm.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    figure1(out, args.source)
    figure2(out, args.source)
    figure3(out, args.source)
    figure4(out, args.source)
    for f in sorted(out.glob("Figure*.png")):
        print(f"{f.name}  {f.stat().st_size/1024:.0f} KB")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
