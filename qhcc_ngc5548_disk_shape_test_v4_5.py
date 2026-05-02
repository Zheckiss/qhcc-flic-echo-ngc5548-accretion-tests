#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qhcc_ngc5548_disk_shape_test_v4_5.py

Shape test for NGC 5548 lag-vs-wavelength structure.

Purpose:
  Show whether selected one-way DCF lag peaks look like:
    (A) disk-only reverberation:
        tau(lambda) = A [(lambda/lambda_ref)^beta - 1]
    or
    (B) FLiC-dominated delay plus a small wavelength-dependent correction:
        tau(lambda) = Delta_t_FLIC + b log(lambda/lambda_pivot)

This script does NOT perform a new echo search.
It consumes the existing v4 ranking table and v4 delay table.

Outputs:
  flic_disk_shape_points_v4_5.csv
  flic_disk_shape_model_comparison_v4_5.csv
  flic_disk_shape_nulls_v4_5.csv
  flic_disk_shape_report_v4_5.md
  flic_disk_shape_comparison_v4_5.png
  flic_disk_shape_residuals_v4_5.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WAVELENGTH_A = {
    "swift_uvw2": 1928.0,
    "swift_uvm2": 2246.0,
    "swift_uvw1": 2600.0,
    "swift_u": 3465.0,
    "swift_b": 4392.0,
    "swift_v": 5468.0,
    "opt_u": 3551.0,
    "opt_b": 4450.0,
    "opt_g": 4770.0,
    "opt_v": 5510.0,
    "opt_r": 6580.0,
    "opt_i": 8060.0,
    "opt_z": 9000.0,
}


def physical_band(response: str) -> str:
    s = str(response).lower()
    s = s.replace("_daily.csv", "")
    s = s.replace(".csv", "")
    s = s.replace("_paper1", "")
    return s


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x * x)))


def fit_disk_only(lambda_A: np.ndarray, tau_h: np.ndarray, beta: float, lambda_ref_A: float = 1367.0) -> Dict:
    x = (lambda_A / lambda_ref_A) ** beta - 1.0
    denom = float(np.sum(x * x))
    if denom <= 0:
        A = 0.0
    else:
        A = float(np.sum(x * tau_h) / denom)
        # Disk-only lag amplitude should not be negative.
        A = max(0.0, A)
    pred = A * x
    return {
        "A": A,
        "beta": beta,
        "rms": rms(tau_h - pred),
        "pred": pred,
    }


def fit_disk_only_free_beta(lambda_A: np.ndarray, tau_h: np.ndarray, beta_min: float, beta_max: float, n_grid: int, lambda_ref_A: float = 1367.0) -> Dict:
    best = None
    for beta in np.linspace(beta_min, beta_max, n_grid):
        cur = fit_disk_only(lambda_A, tau_h, beta, lambda_ref_A=lambda_ref_A)
        if best is None or cur["rms"] < best["rms"]:
            best = cur
    return best


def fit_flic_constant(tau_h: np.ndarray, tau_flic_h: float) -> Dict:
    pred = np.full_like(tau_h, tau_flic_h, dtype=float)
    return {"rms": rms(tau_h - pred), "pred": pred}


def fit_flic_log_slope(lambda_A: np.ndarray, tau_h: np.ndarray, tau_flic_h: float) -> Dict:
    # Use geometric mean pivot. Then the slope term has zero mean in log space.
    # This keeps the central level tied to the fixed FLiC delay.
    loglam = np.log(lambda_A)
    pivot = float(np.exp(np.mean(loglam)))
    x = loglam - math.log(pivot)
    denom = float(np.sum(x * x))
    if denom <= 0:
        b = 0.0
    else:
        b = float(np.sum(x * (tau_h - tau_flic_h)) / denom)
    pred = tau_flic_h + b * x
    return {
        "b_hours_per_log_lambda": b,
        "pivot_A": pivot,
        "rms": rms(tau_h - pred),
        "pred": pred,
    }


def fit_flic_log_slope_with_offset(lambda_A: np.ndarray, tau_h: np.ndarray, tau_flic_h: float) -> Dict:
    # Diagnostic only: lets the layer center float around FLiC.
    loglam = np.log(lambda_A)
    pivot = float(np.exp(np.mean(loglam)))
    x = loglam - math.log(pivot)
    X = np.column_stack([np.ones_like(x), x])
    y = tau_h - tau_flic_h
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = tau_flic_h + a + b * x
    return {
        "offset_h": a,
        "b_hours_per_log_lambda": b,
        "pivot_A": pivot,
        "rms": rms(tau_h - pred),
        "pred": pred,
    }


def load_tau_flic(delays_path: Path, target_name: str) -> float:
    df = pd.read_csv(delays_path)
    if "target_name" in df.columns:
        sub = df[df["target_name"].astype(str) == target_name]
        if len(sub) > 0:
            for col in ["tau_obs_hours", "tau_obs_hours_exact", "delay_obs_hours", "tau_hours"]:
                if col in sub.columns:
                    return float(pd.to_numeric(sub[col], errors="coerce").iloc[0])
    # Flexible fallback: one-way first row.
    for col in ["tau_obs_hours", "tau_obs_hours_exact", "delay_obs_hours", "tau_hours"]:
        if col in df.columns:
            return float(pd.to_numeric(df[col], errors="coerce").iloc[0])
    raise ValueError(f"Could not read FLiC delay from {delays_path}")


def build_points(args: argparse.Namespace) -> pd.DataFrame:
    ranked = pd.read_csv(args.ranked)
    df = ranked.copy()

    if "target_name" not in df.columns:
        raise ValueError("ranked table must contain target_name")
    if "response" not in df.columns:
        raise ValueError("ranked table must contain response")
    if "tau_peak_days" not in df.columns:
        raise ValueError("ranked table must contain tau_peak_days")

    df = df[df["target_name"].astype(str) == args.target_name].copy()
    if "lambda_peak" in df.columns:
        df = df[(df["lambda_peak"] >= args.lambda_min) & (df["lambda_peak"] <= args.lambda_max)].copy()
    if "local_z_peak" in df.columns:
        df = df[pd.to_numeric(df["local_z_peak"], errors="coerce") >= args.min_local_z].copy()

    df["physical_band"] = df["response"].map(physical_band)
    df["wavelength_A"] = df["physical_band"].map(WAVELENGTH_A)
    df = df[np.isfinite(pd.to_numeric(df["wavelength_A"], errors="coerce"))].copy()

    # Keep one row per physical band, the strongest ranked/peak entry.
    if args.unique_physical_bands:
        sort_cols = []
        asc = []
        if "rank_score" in df.columns:
            sort_cols.append("rank_score")
            asc.append(False)
        elif "local_z_peak" in df.columns:
            sort_cols.append("local_z_peak")
            asc.append(False)
        sort_cols.append("response")
        asc.append(True)
        df = df.sort_values(sort_cols, ascending=asc).drop_duplicates("physical_band", keep="first").copy()

    df["tau_peak_hours"] = pd.to_numeric(df["tau_peak_days"], errors="coerce") * 24.0
    df["wavelength_A"] = pd.to_numeric(df["wavelength_A"], errors="coerce")
    df = df[np.isfinite(df["tau_peak_hours"]) & np.isfinite(df["wavelength_A"])].copy()
    df = df.sort_values("wavelength_A").reset_index(drop=True)
    return df


def run_nulls(points: pd.DataFrame, tau_flic_h: float, real_fit: Dict, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    lam = points["wavelength_A"].to_numpy(dtype=float)
    tau = points["tau_peak_hours"].to_numpy(dtype=float)
    real_rms = float(real_fit["rms"])
    real_b = float(real_fit["b_hours_per_log_lambda"])

    rows = []
    for i in range(args.n_null):
        # Permute wavelengths among fixed lags: keeps lag distribution but breaks wavelength ordering.
        lam_perm = rng.permutation(lam)
        fit_perm = fit_flic_log_slope(lam_perm, tau, tau_flic_h)
        rows.append({
            "null_type": "wavelength_permutation",
            "trial": i,
            "rms_flic_log_slope": fit_perm["rms"],
            "slope_b": fit_perm["b_hours_per_log_lambda"],
            "as_good_rms": fit_perm["rms"] <= real_rms,
            "positive_slope": fit_perm["b_hours_per_log_lambda"] >= 0,
            "as_good_and_positive": (fit_perm["rms"] <= real_rms) and (fit_perm["b_hours_per_log_lambda"] >= real_b),
        })

    # Random horizontal center with log-slope, no free offset.
    # This tests how special the pre-fixed FLiC center is among arbitrary centers.
    loglam = np.log(lam)
    pivot = float(np.exp(np.mean(loglam)))
    x = loglam - math.log(pivot)
    denom = float(np.sum(x * x))
    for i in range(args.n_null):
        center = float(rng.uniform(args.random_center_min_hours, args.random_center_max_hours))
        b = float(np.sum(x * (tau - center)) / denom) if denom > 0 else 0.0
        pred = center + b * x
        cur_rms = rms(tau - pred)
        rows.append({
            "null_type": "random_center_log_slope",
            "trial": i,
            "center_hours": center,
            "rms_flic_log_slope": cur_rms,
            "slope_b": b,
            "as_good_rms": cur_rms <= real_rms,
            "positive_slope": b >= 0,
            "as_good_and_positive": (cur_rms <= real_rms) and (b >= real_b),
        })

    return pd.DataFrame(rows)


def plot_comparison(points: pd.DataFrame, tau_flic_h: float, fits: Dict[str, Dict], args: argparse.Namespace, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.4))
    lam = points["wavelength_A"].to_numpy(dtype=float)
    tau = points["tau_peak_hours"].to_numpy(dtype=float)

    ax.scatter(lam, tau, s=70, zorder=5, label="selected one-way DCF peaks")
    for _, row in points.iterrows():
        label = str(row["physical_band"]).replace("swift_", "S:").replace("opt_", "O:")
        ax.annotate(label, (row["wavelength_A"], row["tau_peak_hours"]), xytext=(5, 4), textcoords="offset points", fontsize=8)

    grid = np.linspace(max(1200, lam.min()*0.85), lam.max()*1.05, 300)
    ax.axhline(tau_flic_h, linestyle="--", linewidth=1.5, label=f"FLiC one-way: {tau_flic_h:.2f} h")

    # disk fixed
    beta = fits["disk_fixed"]["beta"]
    A = fits["disk_fixed"]["A"]
    pred_grid = A * ((grid / args.lambda_ref_A) ** beta - 1.0)
    ax.plot(grid, pred_grid, linewidth=2, label=f"disk-only beta=4/3, RMS={fits['disk_fixed']['rms']:.2f} h")

    # disk free
    beta = fits["disk_free"]["beta"]
    A = fits["disk_free"]["A"]
    pred_grid = A * ((grid / args.lambda_ref_A) ** beta - 1.0)
    ax.plot(grid, pred_grid, linewidth=2, label=f"disk-only free beta={beta:.2f}, RMS={fits['disk_free']['rms']:.2f} h")

    # FLiC + log slope
    b = fits["flic_log_slope"]["b_hours_per_log_lambda"]
    pivot = fits["flic_log_slope"]["pivot_A"]
    pred_grid = tau_flic_h + b * (np.log(grid) - math.log(pivot))
    ax.plot(grid, pred_grid, linewidth=2.4, label=f"FLiC + small slope, RMS={fits['flic_log_slope']['rms']:.2f} h")

    ax.set_xlabel("response wavelength, Angstrom")
    ax.set_ylabel("selected lag, hours")
    ax.set_title("NGC 5548: disk-only shape vs FLiC-dominated lag layer")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_residuals(points: pd.DataFrame, tau_flic_h: float, fits: Dict[str, Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    lam = points["wavelength_A"].to_numpy(dtype=float)
    tau = points["tau_peak_hours"].to_numpy(dtype=float)

    res_flic = tau - fits["flic_log_slope"]["pred"]
    res_disk = tau - fits["disk_free"]["pred"]

    ax.scatter(lam, res_flic, s=70, label="residual to FLiC + small slope")
    ax.scatter(lam, res_disk, s=70, marker="x", label="residual to best disk-only")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("response wavelength, Angstrom")
    ax.set_ylabel("residual lag, hours")
    ax.set_title("Residual comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(out_path: Path, points: pd.DataFrame, tau_flic_h: float, model_df: pd.DataFrame, nulls: pd.DataFrame, args: argparse.Namespace) -> None:
    real = model_df.set_index("model")
    p_perm_rms = (1 + int(nulls[(nulls["null_type"] == "wavelength_permutation")]["as_good_rms"].sum())) / (1 + int((nulls["null_type"] == "wavelength_permutation").sum()))
    p_perm_sign = (1 + int(nulls[(nulls["null_type"] == "wavelength_permutation")]["as_good_and_positive"].sum())) / (1 + int((nulls["null_type"] == "wavelength_permutation").sum()))
    p_center = (1 + int(nulls[(nulls["null_type"] == "random_center_log_slope")]["as_good_rms"].sum())) / (1 + int((nulls["null_type"] == "random_center_log_slope").sum()))

    lines = []
    lines.append("# NGC 5548 disk-only shape vs FLiC-dominated lag layer v4.5")
    lines.append("")
    lines.append("This test does not perform a new echo search. It takes the selected one-way DCF peaks from v4")
    lines.append("and asks whether their lag-vs-wavelength shape looks like disk-only reverberation or like")
    lines.append("a fixed FLiC delay plus a small wavelength-dependent correction.")
    lines.append("")
    lines.append("## Selection")
    lines.append("")
    lines.append(f"- Target branch: `{args.target_name}`")
    lines.append(f"- FLiC delay: **{tau_flic_h:.3f} h**")
    lines.append(f"- Lambda window: `{args.lambda_min}` to `{args.lambda_max}`")
    lines.append(f"- Minimum local peak z: `{args.min_local_z}`")
    lines.append(f"- Unique physical bands: `{args.unique_physical_bands}`")
    lines.append(f"- Selected points: **{len(points)}**")
    lines.append("")
    lines.append("## Model comparison")
    lines.append("")
    lines.append(model_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Key result")
    lines.append("")
    if "flic_log_slope" in real.index and "disk_free" in real.index:
        lines.append(f"- RMS for FLiC + small log-slope: **{real.loc['flic_log_slope','rms_hours']:.3f} h**")
        lines.append(f"- RMS for best disk-only curve: **{real.loc['disk_only_free_beta','rms_hours']:.3f} h**")
        lines.append(f"- RMS for disk-only beta=4/3: **{real.loc['disk_only_beta_4over3','rms_hours']:.3f} h**")
        lines.append(f"- Best FLiC-layer slope b: **{real.loc['flic_log_slope','slope_b_hours_per_log_lambda']:.3f} h per log(lambda)**")
    lines.append("")
    lines.append("## Null checks")
    lines.append("")
    lines.append(f"- Wavelength-permutation p(RMS as good as real): **{p_perm_rms:.4f}**")
    lines.append(f"- Wavelength-permutation p(RMS as good and slope at least as positive): **{p_perm_sign:.4f}**")
    lines.append(f"- Random-center p(RMS as good as fixed FLiC center): **{p_center:.4f}**")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("A pure disk-only lag curve is expected to be primarily wavelength-dependent. The FLiC-dominated")
    lines.append("model instead predicts a nearly horizontal layer near the fixed FLiC delay, with a small")
    lines.append("positive wavelength-dependent correction. This script quantifies that shape distinction.")
    lines.append("")
    lines.append("The result should be read as a shape diagnostic, not as a new independent detection claim.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", default="flic_candidates_ranked_v4.csv")
    ap.add_argument("--delays", default="ngc5548_flic_delays_v4.csv")
    ap.add_argument("--out-dir", default="run_v4_5/results")
    ap.add_argument("--target-name", default="one_way_2over3")
    ap.add_argument("--lambda-min", type=float, default=0.75)
    ap.add_argument("--lambda-max", type=float, default=1.30)
    ap.add_argument("--min-local-z", type=float, default=2.5)
    ap.add_argument("--unique-physical-bands", action="store_true", default=True)
    ap.add_argument("--lambda-ref-A", type=float, default=1367.0)
    ap.add_argument("--beta-min", type=float, default=0.3)
    ap.add_argument("--beta-max", type=float, default=3.0)
    ap.add_argument("--beta-grid", type=int, default=271)
    ap.add_argument("--n-null", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--random-center-min-hours", type=float, default=5.0)
    ap.add_argument("--random-center-max-hours", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_path = Path(args.ranked)
    delays_path = Path(args.delays)
    if not ranked_path.exists():
        raise FileNotFoundError(f"Ranked file not found: {ranked_path}")
    if not delays_path.exists():
        raise FileNotFoundError(f"Delay file not found: {delays_path}")

    tau_flic_h = load_tau_flic(delays_path, args.target_name)
    points = build_points(args)
    if len(points) < 3:
        raise RuntimeError("Too few selected points. Lower --min-local-z or broaden --lambda-min/--lambda-max.")

    points.to_csv(out_dir / "flic_disk_shape_points_v4_5.csv", index=False)

    lam = points["wavelength_A"].to_numpy(dtype=float)
    tau = points["tau_peak_hours"].to_numpy(dtype=float)

    fits = {
        "disk_fixed": fit_disk_only(lam, tau, 4.0 / 3.0, lambda_ref_A=args.lambda_ref_A),
        "disk_free": fit_disk_only_free_beta(lam, tau, args.beta_min, args.beta_max, args.beta_grid, lambda_ref_A=args.lambda_ref_A),
        "flic_constant": fit_flic_constant(tau, tau_flic_h),
        "flic_log_slope": fit_flic_log_slope(lam, tau, tau_flic_h),
        "flic_log_slope_offset": fit_flic_log_slope_with_offset(lam, tau, tau_flic_h),
    }

    model_rows = []
    model_rows.append({
        "model": "disk_only_beta_4over3",
        "rms_hours": fits["disk_fixed"]["rms"],
        "A_hours": fits["disk_fixed"]["A"],
        "beta": fits["disk_fixed"]["beta"],
        "slope_b_hours_per_log_lambda": np.nan,
        "offset_hours": np.nan,
    })
    model_rows.append({
        "model": "disk_only_free_beta",
        "rms_hours": fits["disk_free"]["rms"],
        "A_hours": fits["disk_free"]["A"],
        "beta": fits["disk_free"]["beta"],
        "slope_b_hours_per_log_lambda": np.nan,
        "offset_hours": np.nan,
    })
    model_rows.append({
        "model": "flic_constant",
        "rms_hours": fits["flic_constant"]["rms"],
        "A_hours": np.nan,
        "beta": np.nan,
        "slope_b_hours_per_log_lambda": 0.0,
        "offset_hours": 0.0,
    })
    model_rows.append({
        "model": "flic_log_slope",
        "rms_hours": fits["flic_log_slope"]["rms"],
        "A_hours": np.nan,
        "beta": np.nan,
        "slope_b_hours_per_log_lambda": fits["flic_log_slope"]["b_hours_per_log_lambda"],
        "offset_hours": 0.0,
        "pivot_A": fits["flic_log_slope"]["pivot_A"],
    })
    model_rows.append({
        "model": "flic_log_slope_with_offset",
        "rms_hours": fits["flic_log_slope_offset"]["rms"],
        "A_hours": np.nan,
        "beta": np.nan,
        "slope_b_hours_per_log_lambda": fits["flic_log_slope_offset"]["b_hours_per_log_lambda"],
        "offset_hours": fits["flic_log_slope_offset"]["offset_h"],
        "pivot_A": fits["flic_log_slope_offset"]["pivot_A"],
    })
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(out_dir / "flic_disk_shape_model_comparison_v4_5.csv", index=False)

    nulls = run_nulls(points, tau_flic_h, fits["flic_log_slope"], args)
    nulls.to_csv(out_dir / "flic_disk_shape_nulls_v4_5.csv", index=False)

    plot_comparison(points, tau_flic_h, fits, args, out_dir / "flic_disk_shape_comparison_v4_5.png")
    plot_residuals(points, tau_flic_h, fits, out_dir / "flic_disk_shape_residuals_v4_5.png")
    write_report(out_dir / "flic_disk_shape_report_v4_5.md", points, tau_flic_h, model_df, nulls, args)

    print("[done]")
    print(out_dir / "flic_disk_shape_report_v4_5.md")
    print(out_dir / "flic_disk_shape_model_comparison_v4_5.csv")


if __name__ == "__main__":
    main()
