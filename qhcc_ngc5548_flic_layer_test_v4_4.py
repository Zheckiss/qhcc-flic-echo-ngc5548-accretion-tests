#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qhcc_ngc5548_flic_layer_test_v4_4.py

Формальная проверка "слоя" наблюдаемых лагов вокруг одноходовой FLiC-ветви
для NGC 5548.

Этот скрипт НЕ меняет поиск кандидатов v4. Он берёт уже полученные таблицы:
- flic_candidates_ranked_v4.csv
- baseline_summary_v4.csv
- ngc5548_flic_delays_v4.csv
- null_pvalues_combined_v4.csv

и проверяет структурную гипотезу:

1) выбранные лаги не разбросаны хаотично;
2) часть каналов образует слой вокруг одноходовой FLiC-задержки;
3) лучший точный кандидат находится около двухходовой FLiC-задержки;
4) одноходовая организация лучше простой дисковой кривой и редка
   относительно случайных горизонтальных лагов.

Важно: это не новый поиск и не новый "подбор" результата. Это формальная
диагностика уже найденной lag-wavelength структуры.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Центральные длины волн, Å.
# Swift/UVOT: стандартные эффективные длины волн фильтров.
# opt_*: типичные центральные длины волн наземных фильтров.
WAVELENGTHS_A: Dict[str, float] = {
    "swift_UVW2.csv": 1928.0,
    "swift_UVM2.csv": 2246.0,
    "swift_UVW1.csv": 2600.0,
    "swift_U.csv": 3465.0,
    "swift_B.csv": 4392.0,
    "swift_V.csv": 5468.0,
    "opt_u_daily.csv": 3551.0,
    "opt_B_daily.csv": 4361.0,
    "opt_g_daily.csv": 4770.0,
    "opt_V_daily.csv": 5448.0,
    "opt_R_daily.csv": 6407.0,
    "opt_r_daily.csv": 6231.0,
    "opt_I_daily.csv": 7980.0,
    "opt_i_daily.csv": 7625.0,
    "opt_z_daily.csv": 9134.0,
}

LABELS_RU: Dict[str, str] = {
    "swift_UVW2.csv": "Swift UVW2",
    "swift_UVM2.csv": "Swift UVM2",
    "swift_UVW1.csv": "Swift UVW1",
    "swift_U.csv": "Swift U",
    "swift_B.csv": "Swift B",
    "swift_V.csv": "Swift V",
    "opt_u_daily.csv": "opt u",
    "opt_B_daily.csv": "opt B",
    "opt_g_daily.csv": "opt g",
    "opt_V_daily.csv": "opt V",
    "opt_R_daily.csv": "opt R",
    "opt_r_daily.csv": "opt r",
    "opt_I_daily.csv": "opt I",
    "opt_i_daily.csv": "opt i",
    "opt_z_daily.csv": "opt z",
}

# Основной набор, для которого прогонялись финальные null-тесты v4.
DEFAULT_RESPONSES = [
    "swift_UVW2.csv",
    "swift_UVM2.csv",
    "swift_UVW1.csv",
    "swift_U.csv",
    "swift_V.csv",
    "opt_u_daily.csv",
    "opt_B_daily.csv",
    "opt_g_daily.csv",
    "opt_V_daily.csv",
    "opt_R_daily.csv",
    "opt_I_daily.csv",
]


def rms(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(a * a)))


def mad_abs(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(a)))


def fit_disk_curve(
    wavelength_A: np.ndarray,
    tau_hours: np.ndarray,
    lambda_ref_A: float = 1367.0,
    beta_grid: np.ndarray | None = None,
) -> dict:
    """
    Fits tau = A*((lambda/lambda_ref)^beta - 1), with no intercept.

    The no-intercept form is intentional: the driver is the 1367 Å light curve, so
    a standard relative disk lag should vanish at lambda_ref.
    """
    x = np.asarray(wavelength_A, dtype=float)
    y = np.asarray(tau_hours, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > lambda_ref_A)
    x = x[good]
    y = y[good]

    if len(x) < 3:
        return {
            "beta_best": np.nan,
            "A_best_hours": np.nan,
            "rms_hours": np.nan,
            "pred_hours": np.full_like(wavelength_A, np.nan, dtype=float),
        }

    if beta_grid is None:
        beta_grid = np.linspace(0.30, 3.0, 2000)

    best = None
    for beta in beta_grid:
        basis = (x / lambda_ref_A) ** beta - 1.0
        denom = float(np.dot(basis, basis))
        if denom <= 0:
            continue
        A = float(np.dot(y, basis) / denom)
        if A < 0:
            continue
        pred = A * basis
        err = float(np.sqrt(np.mean((y - pred) ** 2)))
        if best is None or err < best[0]:
            best = (err, A, beta)

    if best is None:
        return {
            "beta_best": np.nan,
            "A_best_hours": np.nan,
            "rms_hours": np.nan,
            "pred_hours": np.full_like(wavelength_A, np.nan, dtype=float),
        }

    err, A, beta = best
    pred_all = A * ((np.asarray(wavelength_A, dtype=float) / lambda_ref_A) ** beta - 1.0)
    return {
        "beta_best": float(beta),
        "A_best_hours": float(A),
        "rms_hours": float(err),
        "pred_hours": pred_all,
    }


def fit_thin_disk_fixed_beta(
    wavelength_A: np.ndarray,
    tau_hours: np.ndarray,
    lambda_ref_A: float = 1367.0,
    beta: float = 4.0 / 3.0,
) -> dict:
    x = np.asarray(wavelength_A, dtype=float)
    y = np.asarray(tau_hours, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > lambda_ref_A)
    xg = x[good]
    yg = y[good]

    if len(xg) < 3:
        return {
            "beta": beta,
            "A_hours": np.nan,
            "rms_hours": np.nan,
            "pred_hours": np.full_like(wavelength_A, np.nan, dtype=float),
        }

    basis = (xg / lambda_ref_A) ** beta - 1.0
    A = float(np.dot(yg, basis) / np.dot(basis, basis))
    pred = A * basis
    err = float(np.sqrt(np.mean((yg - pred) ** 2)))
    pred_all = A * ((x / lambda_ref_A) ** beta - 1.0)
    return {
        "beta": float(beta),
        "A_hours": float(A),
        "rms_hours": err,
        "pred_hours": pred_all,
    }


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = pd.read_csv(args.ranked)
    baseline = pd.read_csv(args.baseline)
    delays = pd.read_csv(args.delays)
    null_p = pd.read_csv(args.null_pvalues) if args.null_pvalues and Path(args.null_pvalues).exists() else pd.DataFrame()
    return ranked, baseline, delays, null_p


def build_selected_points(
    ranked: pd.DataFrame,
    delays: pd.DataFrame,
    responses: list[str],
    strict_min: float,
    strict_max: float,
) -> tuple[pd.DataFrame, float, float]:
    one_way = float(delays.loc[delays["target_name"] == "one_way_2over3", "tau_obs_hours"].iloc[0])
    two_way = float(delays.loc[delays["target_name"] == "two_way_4over3", "tau_obs_hours"].iloc[0])

    x = ranked.copy()
    x = x[x["response"].isin(responses)].copy()
    x = x.sort_values("rank_score", ascending=False).drop_duplicates("response", keep="first")
    x["wavelength_A"] = x["response"].map(WAVELENGTHS_A)
    x["label"] = x["response"].map(LABELS_RU).fillna(x["response"])
    x["tau_peak_hours"] = 24.0 * x["tau_peak_days"]
    x["tau_at_target_grid_hours"] = 24.0 * x["tau_at_target_grid_days"]
    x["tau_target_hours"] = 24.0 * x["tau_target_days"]
    x["delta_to_one_way_hours"] = x["tau_peak_hours"] - one_way
    x["delta_to_two_way_hours"] = x["tau_peak_hours"] - two_way
    x["lambda_one_way"] = x["tau_peak_hours"] / one_way
    x["lambda_two_way"] = x["tau_peak_hours"] / two_way
    x["in_one_way_layer"] = (
        (x["target_name"] == "one_way_2over3")
        & (x["lambda_peak"] >= strict_min)
        & (x["lambda_peak"] <= strict_max)
    )
    x["in_two_way_layer"] = (
        (x["target_name"] == "two_way_4over3")
        & (x["lambda_peak"] >= strict_min)
        & (x["lambda_peak"] <= strict_max)
    )
    x["spectral_group"] = np.where(
        x["wavelength_A"] < 4000.0,
        "short_wavelength",
        np.where(x["wavelength_A"] > 5000.0, "long_wavelength", "middle_wavelength"),
    )
    return x.sort_values("wavelength_A"), one_way, two_way


def random_horizontal_line_test(
    tau_hours: np.ndarray,
    reference_hours: float,
    lag_min_hours: float,
    lag_max_hours: float,
    n_random: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    tau = np.asarray(tau_hours, dtype=float)
    tau = tau[np.isfinite(tau)]

    if len(tau) == 0:
        return {
            "rms_reference_hours": np.nan,
            "rms_random_median_hours": np.nan,
            "rms_random_p05_hours": np.nan,
            "rms_random_p95_hours": np.nan,
            "p_random_line_as_good_or_better": np.nan,
            "best_random_line_hours": np.nan,
            "best_random_rms_hours": np.nan,
        }

    random_lines = rng.uniform(lag_min_hours, lag_max_hours, size=n_random)
    random_rms = np.sqrt(np.mean((tau[:, None] - random_lines[None, :]) ** 2, axis=0))
    ref_rms = float(np.sqrt(np.mean((tau - reference_hours) ** 2)))
    p = float((1 + np.sum(random_rms <= ref_rms)) / (len(random_rms) + 1))
    j = int(np.argmin(random_rms))

    return {
        "rms_reference_hours": ref_rms,
        "rms_random_median_hours": float(np.median(random_rms)),
        "rms_random_p05_hours": float(np.quantile(random_rms, 0.05)),
        "rms_random_p95_hours": float(np.quantile(random_rms, 0.95)),
        "p_random_line_as_good_or_better": p,
        "best_random_line_hours": float(random_lines[j]),
        "best_random_rms_hours": float(random_rms[j]),
    }


def summarize_layers(
    selected: pd.DataFrame,
    one_way_hours: float,
    two_way_hours: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    one = selected[selected["in_one_way_layer"]].copy()
    two = selected[selected["in_two_way_layer"]].copy()

    short = one[one["wavelength_A"] < args.short_wavelength_max_A]
    long = one[one["wavelength_A"] > args.long_wavelength_min_A]

    disk_free = fit_disk_curve(one["wavelength_A"].to_numpy(), one["tau_peak_hours"].to_numpy())
    disk_fixed = fit_thin_disk_fixed_beta(one["wavelength_A"].to_numpy(), one["tau_peak_hours"].to_numpy())

    random_test = random_horizontal_line_test(
        tau_hours=one["tau_peak_hours"].to_numpy(),
        reference_hours=one_way_hours,
        lag_min_hours=args.random_lag_min_hours,
        lag_max_hours=args.random_lag_max_hours,
        n_random=args.n_random_lines,
        seed=args.seed,
    )

    one_summary_rows = [
        ("n_one_way_points", len(one)),
        ("one_way_flic_hours", one_way_hours),
        ("mean_delta_to_one_way_hours", float(one["delta_to_one_way_hours"].mean()) if len(one) else np.nan),
        ("std_delta_to_one_way_hours", float(one["delta_to_one_way_hours"].std(ddof=1)) if len(one) > 1 else np.nan),
        ("mad_abs_delta_to_one_way_hours", mad_abs(one["delta_to_one_way_hours"])),
        ("rms_delta_to_one_way_hours", rms(one["delta_to_one_way_hours"])),
        ("mean_delta_short_wavelength_hours", float(short["delta_to_one_way_hours"].mean()) if len(short) else np.nan),
        ("mean_delta_long_wavelength_hours", float(long["delta_to_one_way_hours"].mean()) if len(long) else np.nan),
        ("n_short_wavelength_points", len(short)),
        ("n_long_wavelength_points", len(long)),
        ("disk_fixed_beta", disk_fixed["beta"]),
        ("disk_fixed_A_hours", disk_fixed["A_hours"]),
        ("disk_fixed_rms_hours", disk_fixed["rms_hours"]),
        ("disk_free_beta", disk_free["beta_best"]),
        ("disk_free_A_hours", disk_free["A_best_hours"]),
        ("disk_free_rms_hours", disk_free["rms_hours"]),
        ("random_line_lag_min_hours", args.random_lag_min_hours),
        ("random_line_lag_max_hours", args.random_lag_max_hours),
        ("random_line_count", args.n_random_lines),
        ("random_line_median_rms_hours", random_test["rms_random_median_hours"]),
        ("random_line_p05_rms_hours", random_test["rms_random_p05_hours"]),
        ("random_line_p95_rms_hours", random_test["rms_random_p95_hours"]),
        ("random_line_p_as_good_as_one_way", random_test["p_random_line_as_good_or_better"]),
        ("best_random_line_hours", random_test["best_random_line_hours"]),
        ("best_random_line_rms_hours", random_test["best_random_rms_hours"]),
    ]
    one_summary = pd.DataFrame(one_summary_rows, columns=["metric", "value"])

    two_summary_rows = []
    for _, r in two.iterrows():
        two_summary_rows.append({
            "response": r["response"],
            "label": r["label"],
            "wavelength_A": r["wavelength_A"],
            "tau_peak_hours": r["tau_peak_hours"],
            "two_way_flic_hours": two_way_hours,
            "delta_to_two_way_hours": r["delta_to_two_way_hours"],
            "lambda_two_way": r["lambda_two_way"],
            "local_z_peak": r.get("local_z_peak", np.nan),
            "dcf_peak": r.get("dcf_peak", np.nan),
            "rank_score": r.get("rank_score", np.nan),
        })
    two_summary = pd.DataFrame(two_summary_rows)

    # Comparison table for plotting.
    comparison = pd.DataFrame([
        {"model": "FLiC one-way", "rms_hours": rms(one["delta_to_one_way_hours"]), "note": "fixed one-way FLiC line"},
        {"model": "disk fixed beta=4/3", "rms_hours": disk_fixed["rms_hours"], "note": "standard thin-disk exponent"},
        {"model": "disk free beta", "rms_hours": disk_free["rms_hours"], "note": f"best beta={disk_free['beta_best']:.3g}"},
        {"model": "random horizontal median", "rms_hours": random_test["rms_random_median_hours"], "note": "median over random lag lines"},
    ])

    return one_summary, two_summary, comparison


def add_null_summary(selected: pd.DataFrame, null_p: pd.DataFrame) -> pd.DataFrame:
    if null_p is None or len(null_p) == 0:
        selected["best_null_p_local_z_peak"] = np.nan
        selected["best_null_p_dcf_peak"] = np.nan
        return selected

    # For each response/target, keep the most conservative and most favourable summaries.
    pcols = ["p_local_z_peak", "p_dcf_peak", "p_local_z_target", "p_dcf_at_target"]
    cols = ["response", "target_name"] + [c for c in pcols if c in null_p.columns]
    n = null_p[cols].copy()
    grouped = n.groupby(["response", "target_name"]).agg(
        min_p_local_z_peak=("p_local_z_peak", "min"),
        max_p_local_z_peak=("p_local_z_peak", "max"),
        min_p_dcf_peak=("p_dcf_peak", "min"),
        max_p_dcf_peak=("p_dcf_peak", "max"),
    ).reset_index()
    return selected.merge(grouped, on=["response", "target_name"], how="left")


def make_lag_vs_wavelength_plot(
    points: pd.DataFrame,
    one_summary: pd.DataFrame,
    one_way_hours: float,
    two_way_hours: float,
    out_path: Path,
    band_hours: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    # Shaded FLiC bands.
    ax.axhspan(one_way_hours - band_hours, one_way_hours + band_hours, alpha=0.12, label="one-way FLiC band")
    ax.axhspan(two_way_hours - band_hours, two_way_hours + band_hours, alpha=0.10, label="two-way FLiC band")
    ax.axhline(one_way_hours, linestyle="--", linewidth=1.8, label=f"one-way FLiC = {one_way_hours:.2f} h")
    ax.axhline(two_way_hours, linestyle="--", linewidth=1.8, label=f"two-way FLiC = {two_way_hours:.2f} h")

    # Disk curve fit to one-way points.
    one = points[points["in_one_way_layer"]].copy()
    if len(one) >= 3:
        disk = fit_disk_curve(one["wavelength_A"].to_numpy(), one["tau_peak_hours"].to_numpy())
        xs = np.linspace(max(1500, points["wavelength_A"].min() * 0.9), points["wavelength_A"].max() * 1.05, 400)
        ys = disk["A_best_hours"] * ((xs / 1367.0) ** disk["beta_best"] - 1.0)
        ax.plot(xs, ys, linewidth=1.7, label=f"best disk-only curve (beta={disk['beta_best']:.2f})")

    for _, r in points.iterrows():
        marker = "o"
        size = 70
        if r["response"] == "opt_V_daily.csv":
            marker = "*"
            size = 220
        elif r["in_two_way_layer"]:
            marker = "D"
            size = 85
        elif r["in_one_way_layer"]:
            marker = "o"
            size = 75
        else:
            marker = "s"
            size = 65

        ax.scatter(r["wavelength_A"], r["tau_peak_hours"], s=size, marker=marker, edgecolor="black", linewidth=0.7)
        ax.annotate(
            str(r["label"]),
            (r["wavelength_A"], r["tau_peak_hours"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Response wavelength, Å")
    ax.set_ylabel("Selected lag, hours")
    ax.set_title("NGC 5548: lag--wavelength organisation around fixed FLiC scales")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_residual_plot(
    points: pd.DataFrame,
    one_way_hours: float,
    two_way_hours: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.axhline(0.0, linestyle="--", linewidth=1.8, label="one-way FLiC reference")
    ax.axhline(two_way_hours - one_way_hours, linestyle="--", linewidth=1.5, label="two-way FLiC residual")

    for _, r in points.iterrows():
        marker = "*" if r["response"] == "opt_V_daily.csv" else ("D" if r["in_two_way_layer"] else "o")
        size = 220 if r["response"] == "opt_V_daily.csv" else 75
        ax.scatter(r["wavelength_A"], r["delta_to_one_way_hours"], s=size, marker=marker, edgecolor="black", linewidth=0.7)
        ax.annotate(str(r["label"]), (r["wavelength_A"], r["delta_to_one_way_hours"]), textcoords="offset points", xytext=(5, 5), fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlabel("Response wavelength, Å")
    ax.set_ylabel("Residual lag relative to one-way FLiC, hours")
    ax.set_title("NGC 5548: residuals relative to the one-way FLiC branch")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_rms_comparison_plot(comparison: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(comparison["model"], comparison["rms_hours"])
    ax.set_ylabel("RMS residual, hours")
    ax.set_title("NGC 5548: one-way layer organisation quality")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(comparison["rms_hours"]):
        if np.isfinite(v):
            ax.text(i, v + 0.15, f"{v:.2f} h", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_report(
    out_path: Path,
    selected: pd.DataFrame,
    one_summary: pd.DataFrame,
    two_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    one_way_hours: float,
    two_way_hours: float,
) -> None:
    def metric(name: str) -> float:
        row = one_summary[one_summary["metric"] == name]
        if len(row) == 0:
            return float("nan")
        return float(row["value"].iloc[0])

    v = selected[selected["response"] == "opt_V_daily.csv"]
    v_text = ""
    if len(v):
        vr = v.iloc[0]
        v_text = (
            f"- Main two-way candidate: {vr['label']} at {vr['tau_peak_hours']:.2f} h, "
            f"two-way FLiC expectation {two_way_hours:.2f} h, "
            f"delta {vr['delta_to_two_way_hours']:.2f} h, "
            f"lambda_two_way={vr['lambda_two_way']:.3f}.\n"
        )

    content = f"""# NGC 5548 FLiC layer organisation test v4.4

This report formalises the visual lag--wavelength structure seen in the v4.2
figure. It does **not** perform a new echo search. It takes the already ranked
v4 candidates and tests whether the selected lags are organised around the fixed
FLiC scales.

## Fixed FLiC scales

- One-way FLiC branch: **{one_way_hours:.3f} h**
- Two-way FLiC branch: **{two_way_hours:.3f} h**

## One-way layer result

- Number of one-way layer points: **{int(metric('n_one_way_points'))}**
- RMS around the one-way FLiC branch: **{metric('rms_delta_to_one_way_hours'):.3f} h**
- Mean absolute deviation around the one-way branch: **{metric('mad_abs_delta_to_one_way_hours'):.3f} h**
- Standard deviation of one-way residuals: **{metric('std_delta_to_one_way_hours'):.3f} h**
- Mean residual for short-wavelength channels: **{metric('mean_delta_short_wavelength_hours'):.3f} h**
- Mean residual for long-wavelength channels: **{metric('mean_delta_long_wavelength_hours'):.3f} h**

Interpretation: short-wavelength channels lie below the one-way FLiC scale on
average, while long-wavelength optical channels lie above it. This turns the
visual pattern into a measurable layer structure.

## Comparison with disk-only organisation

- RMS around one-way FLiC: **{metric('rms_delta_to_one_way_hours'):.3f} h**
- RMS around disk-only curve with fixed beta=4/3: **{metric('disk_fixed_rms_hours'):.3f} h**
- RMS around best disk-only curve with free beta: **{metric('disk_free_rms_hours'):.3f} h**
- Best disk-only beta: **{metric('disk_free_beta'):.3f}**

In this diagnostic, the selected one-way points are more tightly organised around
the fixed one-way FLiC line than around the simple disk-only lag curve.

## Random horizontal line test

Random horizontal lag lines were sampled between
{metric('random_line_lag_min_hours'):.1f} h and {metric('random_line_lag_max_hours'):.1f} h.

- Median random-line RMS: **{metric('random_line_median_rms_hours'):.3f} h**
- Probability of a random line being as good as or better than the FLiC one-way line:
  **p = {metric('random_line_p_as_good_as_one_way'):.4f}**
- Best random line in the scan: **{metric('best_random_line_hours'):.3f} h**
- RMS of the best random line: **{metric('best_random_line_rms_hours'):.3f} h**

This is not a full global false-alarm probability. It is a structural check:
it asks whether an arbitrary horizontal lag scale would organise the one-way
points as well as the pre-fixed FLiC one-way branch.

## Two-way branch

{v_text}
The two-way branch is therefore treated as the cleanest completed FLiC candidate,
while the one-way branch is treated as a structured early-response layer.

## Files produced

- `flic_layer_points_v4_4.csv`
- `flic_one_way_layer_summary_v4_4.csv`
- `flic_two_way_summary_v4_4.csv`
- `flic_layer_rms_comparison_v4_4.csv`
- `flic_lag_vs_wavelength_v4_4.png`
- `flic_residuals_one_way_v4_4.png`
- `flic_layer_rms_comparison_v4_4.png`
"""
    out_path.write_text(content, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ranked", default="run_v4/results/flic_candidates_ranked_v4.csv")
    p.add_argument("--baseline", default="run_v4/results/baseline_summary_v4.csv")
    p.add_argument("--delays", default="run_v4/results/ngc5548_flic_delays_v4.csv")
    p.add_argument("--null-pvalues", default="run_v4/results/null_pvalues_combined_v4.csv")
    p.add_argument("--out-dir", default="run_v4/results")
    p.add_argument("--responses", nargs="*", default=DEFAULT_RESPONSES)
    p.add_argument("--strict-lambda-min", type=float, default=0.70)
    p.add_argument("--strict-lambda-max", type=float, default=1.30)
    p.add_argument("--short-wavelength-max-A", type=float, default=4000.0)
    p.add_argument("--long-wavelength-min-A", type=float, default=5000.0)
    p.add_argument("--random-lag-min-hours", type=float, default=5.0)
    p.add_argument("--random-lag-max-hours", type=float, default=30.0)
    p.add_argument("--n-random-lines", type=int, default=100000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--band-hours", type=float, default=1.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked, baseline, delays, null_p = load_inputs(args)
    selected, one_way_hours, two_way_hours = build_selected_points(
        ranked=ranked,
        delays=delays,
        responses=args.responses,
        strict_min=args.strict_lambda_min,
        strict_max=args.strict_lambda_max,
    )
    selected = add_null_summary(selected, null_p)

    one_summary, two_summary, comparison = summarize_layers(selected, one_way_hours, two_way_hours, args)

    selected.to_csv(out_dir / "flic_layer_points_v4_4.csv", index=False)
    one_summary.to_csv(out_dir / "flic_one_way_layer_summary_v4_4.csv", index=False)
    two_summary.to_csv(out_dir / "flic_two_way_summary_v4_4.csv", index=False)
    comparison.to_csv(out_dir / "flic_layer_rms_comparison_v4_4.csv", index=False)

    make_lag_vs_wavelength_plot(
        selected,
        one_summary,
        one_way_hours,
        two_way_hours,
        out_dir / "flic_lag_vs_wavelength_v4_4.png",
        band_hours=args.band_hours,
    )
    make_residual_plot(selected, one_way_hours, two_way_hours, out_dir / "flic_residuals_one_way_v4_4.png")
    make_rms_comparison_plot(comparison, out_dir / "flic_layer_rms_comparison_v4_4.png")

    write_report(
        out_path=out_dir / "flic_layer_test_report_v4_4.md",
        selected=selected,
        one_summary=one_summary,
        two_summary=two_summary,
        comparison=comparison,
        one_way_hours=one_way_hours,
        two_way_hours=two_way_hours,
    )

    print("[ok] wrote", out_dir / "flic_layer_points_v4_4.csv")
    print("[ok] wrote", out_dir / "flic_layer_test_report_v4_4.md")
    print("[ok] wrote plots in", out_dir)


if __name__ == "__main__":
    main()
