#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qhcc_ngc5548_core_v4.py

QHCC/FLiC accretion-echo pilot core for NGC 5548.

v4 changes relative to v3:
1. Target FLiC delays are computed from config: mass, redshift and branch list.
   They are no longer hard-coded as TAU_PRIMARY / TAU_DOUBLE in the search.
2. Default scientific branches are alpha=2/3 and alpha=4/3 only.
   The old alpha=3/4 diagnostic is not a default science channel.
3. Baseline, manual lag-pair and null tests all receive the same target list.
4. Added false-lambda null mode: tests whether comparable peaks appear away
   from the predeclared lambda≈1 region.
5. Added a baseline candidate aggregator for compact reporting.

The heavy data-preparation utilities are reused from qhcc_ngc5548_core_v3.py.
Keep both files in the same directory, or vendor the v3 preparation functions into
this file if a fully standalone release is needed.
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

from qhcc_ngc5548_core_v3 import (
    # data / preparation layer
    ensure_dirs,
    download_all,
    prepare_curves,
    write_manifest,
    load_curve,
    # constants and DCF helpers
    G,
    C,
    M_SUN,
    L_PLANCK,
    DAY,
    precompute_pair,
    dcf_score_from_response,
    summarize_target,
    circular_shift,
    permute_flux,
    ou_like_surrogate,
    empirical_p_value,
)


DEFAULT_BRANCHES = [
    {"name": "one_way_2over3", "alpha": 2.0 / 3.0, "label": "one-way, alpha=2/3"},
    {"name": "two_way_4over3", "alpha": 4.0 / 3.0, "label": "two-way, alpha=4/3"},
]


def flic_delay_days(mass_msun: float, z: float, alpha: float) -> tuple[float, float]:
    """Return source-frame and observer-frame FLiC delay in days."""
    rs = 2.0 * G * float(mass_msun) * M_SUN / C**2
    tau_source = float(alpha) * (rs / C) * math.log(rs / L_PLANCK) / DAY
    tau_obs = (1.0 + float(z)) * tau_source
    return tau_source, tau_obs


def normalise_branches(branches: list[dict] | None = None) -> list[dict]:
    """Validate branches from config and fall back to strict QHCC defaults."""
    if not branches:
        branches = DEFAULT_BRANCHES
    out = []
    for i, br in enumerate(branches):
        alpha = float(br.get("alpha"))
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid branch alpha at index {i}: {br}")
        name = str(br.get("name") or f"alpha_{alpha:.6g}")
        label = str(br.get("label") or name)
        out.append({"name": name, "alpha": alpha, "label": label})
    return out


def build_flic_targets(mass_msun: float, z: float, branches: list[dict] | None = None) -> list[dict]:
    """Build all FLiC search targets from mass, redshift and branch definitions."""
    targets = []
    for br in normalise_branches(branches):
        src, obs = flic_delay_days(float(mass_msun), float(z), float(br["alpha"]))
        targets.append({
            "target_name": br["name"],
            "branch_label": br.get("label", br["name"]),
            "mass_msun": float(mass_msun),
            "z": float(z),
            "alpha": float(br["alpha"]),
            "tau_source_days": float(src),
            "tau_source_hours": float(src * 24.0),
            "tau_obs_days": float(obs),
            "tau_obs_hours": float(obs * 24.0),
        })
    return targets


def build_targets_from_config(cfg: dict) -> list[dict]:
    return build_flic_targets(
        mass_msun=float(cfg["mass_msun"]),
        z=float(cfg.get("redshift", 0.0)),
        branches=cfg.get("branches"),
    )


def write_delay_table(
    results_dir: Path,
    mass_msun: float = 6.5e7,
    z: float = 0.017175,
    branches: list[dict] | None = None,
    targets: list[dict] | None = None,
) -> pd.DataFrame:
    """Save the FLiC delay table used by v4."""
    if targets is None:
        targets = build_flic_targets(mass_msun, z, branches=branches)
    df = pd.DataFrame(targets)
    if len(df):
        df["two_tau_obs_days"] = 2.0 * df["tau_obs_days"]
        df["two_tau_obs_hours"] = 2.0 * df["tau_obs_hours"]
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "ngc5548_flic_delays_v4.csv", index=False)
    return df


def _target_summary_rows(ccf: pd.DataFrame, targets: list[dict], peak_window: float, side_min: float, side_max: float) -> list[dict]:
    rows = []
    for t in targets:
        row = summarize_target(
            ccf,
            float(t["tau_obs_days"]),
            str(t["target_name"]),
            peak_window,
            side_min,
            side_max,
        )
        row.update({
            "branch_label": t.get("branch_label"),
            "alpha": t.get("alpha"),
            "mass_msun": t.get("mass_msun"),
            "z": t.get("z"),
            "tau_source_days": t.get("tau_source_days"),
            "tau_source_hours": t.get("tau_source_hours"),
            "tau_obs_days_exact": t.get("tau_obs_days"),
            "tau_obs_hours_exact": t.get("tau_obs_hours"),
        })
        rows.append(row)
    return rows


def run_lag_pair(
    driver_path: Path,
    response_path: Path,
    out_prefix: Path,
    targets: list[dict] | None = None,
    lag_min: float = -2.0,
    lag_max: float = 4.0,
    lag_step: float = 0.01,
    bin_width: float = 0.08,
    peak_window: float = 0.15,
    side_min: float = 0.25,
    side_max: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if targets is None:
        targets = build_flic_targets(6.5e7, 0.017175)

    driver = load_curve(driver_path)
    response = load_curve(response_path)
    pre = precompute_pair(driver, response)
    lags = np.arange(lag_min, lag_max + 0.5 * lag_step, lag_step)

    ccf = dcf_score_from_response(pre, response["flux"].to_numpy(float), lags, bin_width)
    rows = _target_summary_rows(ccf, targets, peak_window, side_min, side_max)

    summary = pd.DataFrame(rows)
    summary.insert(0, "driver", driver_path.name)
    summary.insert(1, "response", response_path.name)
    summary.insert(2, "bin_width_days", bin_width)
    summary.insert(3, "peak_window_days", peak_window)
    summary.insert(4, "lag_min_days", lag_min)
    summary.insert(5, "lag_max_days", lag_max)
    summary.insert(6, "lag_step_days", lag_step)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ccf.to_csv(Path(str(out_prefix) + ".ccf.csv"), index=False)
    summary.to_csv(Path(str(out_prefix) + ".summary.csv"), index=False)
    return ccf, summary


def run_baseline(
    curves_dir: Path,
    results_dir: Path,
    driver_name: str,
    response_names: list[str],
    targets: list[dict] | None = None,
    bin_width: float = 0.08,
    peak_window: float = 0.15,
    lag_min: float = -2.0,
    lag_max: float = 4.0,
    lag_step: float = 0.01,
    side_min: float = 0.25,
    side_max: float = 3.0,
) -> pd.DataFrame:
    driver_path = curves_dir / driver_name
    if not driver_path.exists():
        raise FileNotFoundError(f"No driver curve: {driver_path}")

    summaries = []
    for resp_name in response_names:
        resp_path = curves_dir / resp_name
        if not resp_path.exists():
            print(f"[skip] missing response: {resp_path}")
            continue
        out_prefix = results_dir / f"{driver_path.stem}_TO_{resp_path.stem}_v4"
        _, s = run_lag_pair(
            driver_path,
            resp_path,
            out_prefix,
            targets=targets,
            lag_min=lag_min,
            lag_max=lag_max,
            lag_step=lag_step,
            bin_width=bin_width,
            peak_window=peak_window,
            side_min=side_min,
            side_max=side_max,
        )
        summaries.append(s)

    out = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    out.to_csv(results_dir / "baseline_summary_v4.csv", index=False)
    return out


def summarize_from_flux(
    pre,
    response_flux: np.ndarray,
    lags: np.ndarray,
    bin_width: float,
    peak_window: float,
    side_min: float,
    side_max: float,
    targets: list[dict],
) -> list[dict]:
    ccf = dcf_score_from_response(pre, response_flux, lags, bin_width)
    return _target_summary_rows(ccf, targets, peak_window, side_min, side_max)


def _sample_false_lambda(rng: np.random.Generator, strict_lambda_min: float, strict_lambda_max: float) -> float:
    ranges = []
    if strict_lambda_min > 0.25:
        ranges.append((0.25, strict_lambda_min))
    if strict_lambda_max < 2.50:
        ranges.append((strict_lambda_max, 2.50))
    if not ranges:
        ranges = [(0.25, 0.70), (1.30, 2.50)]
    a, b = ranges[int(rng.integers(0, len(ranges)))]
    return float(rng.uniform(a, b))


def run_null_tests(
    curves_dir: Path,
    results_dir: Path,
    pair: str,
    targets: list[dict] | None = None,
    n_null: int = 1000,
    modes: list[str] | None = None,
    seed: int = 12345,
    lag_min: float = -2.0,
    lag_max: float = 4.0,
    lag_step: float = 0.01,
    bin_width: float = 0.08,
    peak_window: float = 0.15,
    side_min: float = 0.25,
    side_max: float = 3.0,
    strict_lambda_min: float = 0.7,
    strict_lambda_max: float = 1.3,
) -> dict[str, pd.DataFrame]:
    if targets is None:
        targets = build_flic_targets(6.5e7, 0.017175)
    if modes is None:
        modes = ["shift", "permute", "ou", "false_lambda"]
    if ":" not in pair:
        raise ValueError("--null-pair must be driver.csv:response.csv")

    driver_name, response_name = pair.split(":", 1)
    driver_path = curves_dir / driver_name
    response_path = curves_dir / response_name

    driver = load_curve(driver_path)
    response = load_curve(response_path)
    pre = precompute_pair(driver, response)
    y = response["flux"].to_numpy(float)
    t = response["time"].to_numpy(float)
    lags = np.arange(lag_min, lag_max + 0.5 * lag_step, lag_step)

    real_rows = summarize_from_flux(pre, y, lags, bin_width, peak_window, side_min, side_max, targets)
    real_df = pd.DataFrame(real_rows)
    real_df.insert(0, "driver", driver_name)
    real_df.insert(1, "response", response_name)
    real_df.insert(2, "bin_width_days", bin_width)
    real_df.insert(3, "peak_window_days", peak_window)

    if real_df["n_pairs_at_peak"].fillna(0).sum() == 0:
        raise RuntimeError("No data pairs near target lags. Check time scale, bin width, or lag range.")

    real_ccf = dcf_score_from_response(pre, y, lags, bin_width)
    rng = np.random.default_rng(seed)
    null_rows = []

    for mode in modes:
        print(f"[null] {pair} mode={mode}")
        for i in range(n_null):
            if mode == "false_lambda":
                for target in targets:
                    lam = _sample_false_lambda(rng, strict_lambda_min, strict_lambda_max)
                    false_tau = float(target["tau_obs_days"]) * lam
                    if false_tau < lag_min or false_tau > lag_max:
                        continue
                    r = summarize_target(real_ccf, false_tau, str(target["target_name"]), peak_window, side_min, side_max)
                    r.update({
                        "branch_label": target.get("branch_label"),
                        "alpha": target.get("alpha"),
                        "mass_msun": target.get("mass_msun"),
                        "z": target.get("z"),
                        "tau_source_days": target.get("tau_source_days"),
                        "tau_obs_days_exact": target.get("tau_obs_days"),
                        "false_lambda": lam,
                        "false_tau_days": false_tau,
                    })
                    r["null_mode"] = mode
                    r["null_index"] = i
                    null_rows.append(r)
            else:
                if mode == "shift":
                    yy = circular_shift(y, rng)
                elif mode == "permute":
                    yy = permute_flux(y, rng)
                elif mode == "ou":
                    yy = ou_like_surrogate(t, y, rng)
                else:
                    raise ValueError(f"Unknown null mode: {mode}")
                for r in summarize_from_flux(pre, yy, lags, bin_width, peak_window, side_min, side_max, targets):
                    r["null_mode"] = mode
                    r["null_index"] = i
                    null_rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n_null}")

    null_df = pd.DataFrame(null_rows)

    p_rows = []
    for _, real in real_df.iterrows():
        target_name = real["target_name"]
        for mode in modes:
            sub = null_df[(null_df["target_name"] == target_name) & (null_df["null_mode"] == mode)]
            p_rows.append({
                "driver": driver_name,
                "response": response_name,
                "target_name": target_name,
                "branch_label": real.get("branch_label"),
                "alpha": real.get("alpha"),
                "tau_obs_days_exact": real.get("tau_obs_days_exact"),
                "null_mode": mode,
                "real_local_z_target": real["local_z_target"],
                "real_local_z_peak": real["local_z_peak"],
                "real_dcf_at_target": real["dcf_at_target"],
                "real_dcf_peak": real["dcf_peak"],
                "real_lambda_peak": real["lambda_peak"],
                "p_local_z_target": empirical_p_value(sub["local_z_target"].to_numpy(float), real["local_z_target"]),
                "p_local_z_peak": empirical_p_value(sub["local_z_peak"].to_numpy(float), real["local_z_peak"]),
                "p_dcf_at_target": empirical_p_value(sub["dcf_at_target"].to_numpy(float), real["dcf_at_target"]),
                "p_dcf_peak": empirical_p_value(sub["dcf_peak"].to_numpy(float), real["dcf_peak"]),
                "null_local_z_peak_mean": float(np.nanmean(sub["local_z_peak"])) if len(sub) else np.nan,
                "null_local_z_peak_std": float(np.nanstd(sub["local_z_peak"], ddof=1)) if len(sub) > 1 else np.nan,
                "null_local_z_peak_max": float(np.nanmax(sub["local_z_peak"])) if len(sub) else np.nan,
            })

    pval_df = pd.DataFrame(p_rows)
    if len(null_df):
        null_summary = null_df.groupby(["null_mode", "target_name"]).agg(
            count=("local_z_peak", "count"),
            local_z_target_mean=("local_z_target", "mean"),
            local_z_target_std=("local_z_target", "std"),
            local_z_target_max=("local_z_target", "max"),
            local_z_peak_mean=("local_z_peak", "mean"),
            local_z_peak_std=("local_z_peak", "std"),
            local_z_peak_max=("local_z_peak", "max"),
        ).reset_index()
    else:
        null_summary = pd.DataFrame()

    safe_pair = f"{Path(driver_name).stem}_TO_{Path(response_name).stem}"
    prefix = results_dir / f"null_{safe_pair}_v4"
    real_df.to_csv(Path(str(prefix) + ".real.csv"), index=False)
    null_df.to_csv(Path(str(prefix) + ".distribution.csv"), index=False)
    null_summary.to_csv(Path(str(prefix) + ".summary.csv"), index=False)
    pval_df.to_csv(Path(str(prefix) + ".pvalues.csv"), index=False)

    return {
        "real": real_df,
        "distribution": null_df,
        "summary": null_summary,
        "pvalues": pval_df,
    }


def aggregate_baseline_candidates(
    results_dir: Path,
    strict_lambda_min: float = 0.7,
    strict_lambda_max: float = 1.3,
) -> pd.DataFrame:
    """Create a ranked compact candidate table from baseline_summary_v4.csv."""
    path = results_dir / "baseline_summary_v4.csv"
    if not path.exists():
        raise FileNotFoundError(f"No baseline summary found: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        out = pd.DataFrame()
        out.to_csv(results_dir / "flic_candidates_ranked_v4.csv", index=False)
        return out

    for col in ["local_z_target", "local_z_peak", "dcf_at_target", "dcf_peak", "lambda_peak", "lambda_target_grid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["lambda_peak_in_strict_band"] = (df["lambda_peak"] >= strict_lambda_min) & (df["lambda_peak"] <= strict_lambda_max)
    df["lambda_target_in_strict_band"] = (df["lambda_target_grid"] >= strict_lambda_min) & (df["lambda_target_grid"] <= strict_lambda_max)
    # Primary score is target-dominated; peak is supportive but downweighted.
    df["rank_score"] = df["local_z_target"].fillna(-999.0) + 0.25 * df["local_z_peak"].fillna(-999.0)
    df = df.sort_values(
        ["lambda_peak_in_strict_band", "rank_score", "dcf_at_target"],
        ascending=[False, False, False],
    )
    df.to_csv(results_dir / "flic_candidates_ranked_v4.csv", index=False)
    return df
