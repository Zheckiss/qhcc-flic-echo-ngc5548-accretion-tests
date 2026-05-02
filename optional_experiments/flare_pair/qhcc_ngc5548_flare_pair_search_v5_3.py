#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qhcc_ngc5548_flare_pair_search_v5_3.py

All-band flare-pair QHCC/FLiC search for NGC 5548.

Version 5.3 adds duplicate-curve control: --exclude patterns and
--unique-physical-bands to avoid counting the same physical band twice
(for example hst_cos_1367.csv and hst_cos_1367_paper1.csv).

This script does NOT assume that one photometric band physically drives another.
It treats every significant peak in every selected light curve as a possible
accretion-flare marker, then asks whether flare pairs across all bands are
preferentially separated by the fixed FLiC delays.

Core test:
    count flare pairs with |dt - tau_FLiC| <= tolerance
    compare with false-lag and random-time nulls.

Inputs:
    - run_v4/curves/*.csv               light curves with columns time, flux, flux_err
    - run_v4/results/ngc5548_flic_delays_v4.csv

Outputs:
    - flare_catalog_v5.csv
    - flare_pairs_all_v5.csv
    - flare_pairs_flic_hits_v5.csv
    - flare_pair_summary_v5.csv
    - flare_pair_null_distribution_v5.csv
    - flare_pair_report_v5.md
    - flare_pair_lag_histogram_v5.png
    - flare_pair_timeline_v5.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Built-in NGC 5548 FLiC delay fallback
# -----------------------------------------------------------------------------

G_SI = 6.67430e-11
C_SI = 299_792_458.0
M_SUN_SI = 1.98847e30
L_PLANCK_SI = 1.616255e-35
DAY_S = 86400.0


def flic_delay_hours(mass_msun: float, z: float, alpha: float) -> tuple[float, float]:
    """Return source-frame and observer-frame FLiC delay in hours."""
    rs = 2.0 * G_SI * mass_msun * M_SUN_SI / (C_SI ** 2)
    tau_source_h = alpha * (rs / C_SI) * math.log(rs / L_PLANCK_SI) / 3600.0
    tau_obs_h = (1.0 + z) * tau_source_h
    return tau_source_h, tau_obs_h


def default_ngc5548_flic_targets(mass_msun: float = 6.5e7, z: float = 0.017175) -> pd.DataFrame:
    """Built-in fallback targets, used when no delay CSV is found."""
    rows = []
    for target_name, branch_label, alpha in [
        ("one_way_2over3", "one-way, alpha=2/3", 2.0 / 3.0),
        ("two_way_4over3", "two-way, alpha=4/3", 4.0 / 3.0),
    ]:
        src_h, obs_h = flic_delay_hours(mass_msun, z, alpha)
        rows.append({
            "target_name": target_name,
            "branch_label": branch_label,
            "mass_msun": mass_msun,
            "z": z,
            "alpha": alpha,
            "tau_source_hours": src_h,
            "tau_obs_hours": obs_h,
            "tau_source_days": src_h / 24.0,
            "tau_obs_days": obs_h / 24.0,
        })
    return pd.DataFrame(rows)


def resolve_delay_file(requested: str | None, curves_dir: Path) -> Path | None:
    """Find delay CSV in common locations; return None if not found."""
    candidates = []
    if requested:
        candidates.append(Path(requested))
    candidates.extend([
        Path("ngc5548_flic_delays_v4.csv"),
        Path("results") / "ngc5548_flic_delays_v4.csv",
        Path("run_v4") / "results" / "ngc5548_flic_delays_v4.csv",
        curves_dir.parent / "results" / "ngc5548_flic_delays_v4.csv",
        curves_dir.parent / "ngc5548_flic_delays_v4.csv",
        curves_dir.parent.parent / "results" / "ngc5548_flic_delays_v4.csv",
        curves_dir.parent.parent / "run_v4" / "results" / "ngc5548_flic_delays_v4.csv",
    ])
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            return p
    return None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -----------------------------------------------------------------------------
# Band metadata
# -----------------------------------------------------------------------------

BAND_WAVELENGTH_A = {
    # HST/COS continua
    "hst_cos_1158": 1158.0,
    "hst_cos_1367": 1367.0,
    "hst_cos_1479": 1479.0,
    "hst_cos_1746": 1746.0,
    "hst_cos_1825": 1825.0,
    # Swift/UVOT central wavelengths, approximate
    "swift_UVW2": 1928.0,
    "swift_UVM2": 2246.0,
    "swift_UVW1": 2600.0,
    "swift_U": 3465.0,
    "swift_B": 4392.0,
    "swift_V": 5468.0,
    # Ground optical, approximate effective wavelengths
    "opt_u": 3540.0,
    "opt_B": 4380.0,
    "opt_g": 4770.0,
    "opt_V": 5450.0,
    "opt_r": 6231.0,
    "opt_R": 6410.0,
    "opt_i": 7625.0,
    "opt_I": 7980.0,
    "opt_z": 9134.0,
    # XRT approximate representative labels in Angstrom are not physically useful.
    "swift_xrt_soft": np.nan,
    "swift_xrt_hard": np.nan,
}


def canonical_curve_id(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    stem = re.sub(r"_daily$", "", stem)
    stem = re.sub(r"_raw$", "", stem)
    stem = re.sub(r"_paper1$", "", stem)
    stem = re.sub(r"_0p3_0p8keV$", "", stem)
    stem = re.sub(r"_0p8_10keV$", "", stem)
    return stem


def curve_class(curve_id: str) -> str:
    if curve_id.startswith("line_"):
        return "line"
    if curve_id.startswith("swift_xrt"):
        return "xray"
    return "continuum"


def wavelength_for_curve(curve_file: str) -> float:
    cid = canonical_curve_id(curve_file)
    if cid in BAND_WAVELENGTH_A:
        return float(BAND_WAVELENGTH_A[cid])
    m = re.match(r"hst_cos_(\d+)", cid)
    if m:
        return float(m.group(1))
    return np.nan


# -----------------------------------------------------------------------------
# IO and preprocessing
# -----------------------------------------------------------------------------


def load_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"time", "flux", "flux_err"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df[["time", "flux", "flux_err"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 0:
        sig = np.nanstd(x, ddof=1)
    if not np.isfinite(sig) or sig <= 0:
        sig = 1.0
    return float(sig)


def rolling_median_irregular(t: np.ndarray, y: np.ndarray, window_days: float) -> np.ndarray:
    """Simple robust trend estimate for irregular light curves."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    half = 0.5 * float(window_days)
    out = np.empty_like(y, dtype=float)
    for i, ti in enumerate(t):
        m = (t >= ti - half) & (t <= ti + half) & np.isfinite(y)
        if m.sum() >= 5:
            out[i] = np.nanmedian(y[m])
        elif np.isfinite(y).sum() > 0:
            out[i] = np.nanmedian(y[np.isfinite(y)])
        else:
            out[i] = np.nan
    return out


@dataclass
class FlareDetectConfig:
    min_z: float = 2.0
    trend_window_days: float = 5.0
    min_separation_hours: float = 6.0
    max_flares_per_curve: int = 50
    include_negative: bool = False


def detect_flares(curve_path: Path, cfg: FlareDetectConfig) -> pd.DataFrame:
    df = load_curve(curve_path)
    if len(df) < 5:
        return pd.DataFrame()

    t = df["time"].to_numpy(float)
    y = df["flux"].to_numpy(float)
    err = df["flux_err"].to_numpy(float)

    trend = rolling_median_irregular(t, y, cfg.trend_window_days)
    resid = y - trend
    sig_intr = robust_sigma(resid)
    err_med = np.nanmedian(err[np.isfinite(err) & (err > 0)]) if np.any(np.isfinite(err) & (err > 0)) else 0.0
    scale = math.sqrt(sig_intr * sig_intr + float(err_med) * float(err_med))
    if not np.isfinite(scale) or scale <= 0:
        scale = sig_intr if sig_intr > 0 else 1.0
    z = resid / scale

    candidate_idx: list[int] = []
    for i in range(len(df)):
        zi = z[i]
        if not np.isfinite(zi):
            continue
        if cfg.include_negative:
            pass_thresh = abs(zi) >= cfg.min_z
            better_left = i == 0 or abs(zi) >= abs(z[i - 1])
            better_right = i == len(df) - 1 or abs(zi) >= abs(z[i + 1])
        else:
            pass_thresh = zi >= cfg.min_z
            better_left = i == 0 or zi >= z[i - 1]
            better_right = i == len(df) - 1 or zi >= z[i + 1]
        if pass_thresh and better_left and better_right:
            candidate_idx.append(i)

    # Consolidate peaks that are too close in time, keeping the strongest by |z|.
    min_sep_days = cfg.min_separation_hours / 24.0
    candidate_idx = sorted(candidate_idx, key=lambda i: t[i])
    selected: list[int] = []
    cluster: list[int] = []

    def flush_cluster(cl: list[int]) -> None:
        if not cl:
            return
        best = max(cl, key=lambda k: abs(z[k]) if cfg.include_negative else z[k])
        selected.append(best)

    last_t = None
    for idx in candidate_idx:
        if last_t is None or (t[idx] - last_t) <= min_sep_days:
            cluster.append(idx)
        else:
            flush_cluster(cluster)
            cluster = [idx]
        last_t = t[idx]
    flush_cluster(cluster)

    selected = sorted(selected, key=lambda i: abs(z[i]) if cfg.include_negative else z[i], reverse=True)
    if cfg.max_flares_per_curve > 0:
        selected = selected[: cfg.max_flares_per_curve]
    selected = sorted(selected, key=lambda i: t[i])

    cid = canonical_curve_id(curve_path.name)
    rows = []
    for j, i in enumerate(selected):
        rows.append({
            "curve_file": curve_path.name,
            "curve_id": cid,
            "curve_class": curve_class(cid),
            "wavelength_A": wavelength_for_curve(curve_path.name),
            "flare_index_in_curve": j,
            "time_days": float(t[i]),
            "time_hours": float(t[i] * 24.0),
            "flux": float(y[i]),
            "flux_err": float(err[i]) if np.isfinite(err[i]) else np.nan,
            "trend": float(trend[i]) if np.isfinite(trend[i]) else np.nan,
            "residual": float(resid[i]),
            "z_score": float(z[i]),
            "abs_z_score": float(abs(z[i])),
            "detection_scale": float(scale),
        })
    return pd.DataFrame(rows)


def curve_preference_score(p: Path) -> int:
    """Lower score is preferred when two files represent the same physical band."""
    stem = p.stem
    score = 0
    # Prefer canonical processed files over legacy duplicate exports.
    if stem.endswith("_paper1"):
        score += 100
    # Prefer binned daily optical files over raw files when both exist.
    if stem.endswith("_raw"):
        score += 50
    if stem.endswith("_daily"):
        score -= 10
    return score


def matches_exclude(p: Path, patterns: list[str]) -> bool:
    if not patterns:
        return False
    cid = canonical_curve_id(p.name)
    for pat in patterns:
        pat = str(pat).strip()
        if not pat or pat.startswith("#"):
            continue
        if fnmatch.fnmatch(p.name, pat) or fnmatch.fnmatch(p.stem, pat) or fnmatch.fnmatch(cid, pat):
            return True
    return False


def load_exclude_patterns(args: argparse.Namespace) -> list[str]:
    patterns = list(args.exclude or [])
    if args.exclude_file:
        p = Path(args.exclude_file)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
        else:
            print(f"[warn] exclude file not found: {p}")
    return patterns


def discover_curve_files(
    curves_dir: Path,
    include_lines: bool,
    include_xray: bool,
    pattern: str = "*.csv",
    exclude_patterns: list[str] | None = None,
    unique_physical_bands: bool = False,
) -> list[Path]:
    files = sorted(curves_dir.glob(pattern))
    exclude_patterns = exclude_patterns or []
    out = []
    skipped_rows = []
    for p in files:
        cid = canonical_curve_id(p.name)
        cls = curve_class(cid)
        if matches_exclude(p, exclude_patterns):
            skipped_rows.append((p.name, cid, "exclude_pattern"))
            print(f"[skip exclude] {p.name}")
            continue
        if cls == "line" and not include_lines:
            skipped_rows.append((p.name, cid, "line_disabled"))
            continue
        if cls == "xray" and not include_xray:
            skipped_rows.append((p.name, cid, "xray_disabled"))
            continue
        # Skip raw optical if daily version exists to avoid duplicate time series.
        if p.stem.endswith("_raw") and (p.parent / (p.stem.replace("_raw", "_daily") + ".csv")).exists():
            skipped_rows.append((p.name, cid, "raw_daily_duplicate"))
            continue
        out.append(p)

    if unique_physical_bands:
        best: dict[str, Path] = {}
        for p in out:
            cid = canonical_curve_id(p.name)
            if cid not in best:
                best[cid] = p
            else:
                old = best[cid]
                if curve_preference_score(p) < curve_preference_score(old):
                    print(f"[skip duplicate] {old.name} replaced by {p.name} for {cid}")
                    skipped_rows.append((old.name, cid, f"duplicate_replaced_by:{p.name}"))
                    best[cid] = p
                else:
                    print(f"[skip duplicate] {p.name} kept {old.name} for {cid}")
                    skipped_rows.append((p.name, cid, f"duplicate_kept:{old.name}"))
        out = sorted(best.values(), key=lambda x: x.name)

    # Attach a lightweight audit trail for build_flare_catalog.
    discover_curve_files.last_skipped_rows = skipped_rows  # type: ignore[attr-defined]
    return out


def build_flare_catalog(curves_dir: Path, out_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    exclude_patterns = load_exclude_patterns(args)
    files = discover_curve_files(
        curves_dir,
        include_lines=args.include_lines,
        include_xray=args.include_xray,
        pattern=args.curve_glob,
        exclude_patterns=exclude_patterns,
        unique_physical_bands=args.unique_physical_bands,
    )
    inventory_rows = []
    skipped_rows = getattr(discover_curve_files, "last_skipped_rows", [])
    for p in files:
        cid = canonical_curve_id(p.name)
        inventory_rows.append({
            "curve_file": p.name,
            "curve_id": cid,
            "curve_class": curve_class(cid),
            "wavelength_A": wavelength_for_curve(p.name),
            "status": "used",
            "reason": "",
        })
    for name, cid, reason in skipped_rows:
        inventory_rows.append({
            "curve_file": name,
            "curve_id": cid,
            "curve_class": curve_class(cid),
            "wavelength_A": wavelength_for_curve(name),
            "status": "skipped",
            "reason": reason,
        })
    pd.DataFrame(inventory_rows).sort_values(["status", "curve_file"]).to_csv(out_dir / "curve_inventory_v5_3.csv", index=False)
    cfg = FlareDetectConfig(
        min_z=args.min_z,
        trend_window_days=args.trend_window_days,
        min_separation_hours=args.min_separation_hours,
        max_flares_per_curve=args.max_flares_per_curve,
        include_negative=args.include_negative,
    )
    frames = []
    for p in files:
        try:
            fl = detect_flares(p, cfg)
        except Exception as exc:
            print(f"[warn] skipped {p.name}: {exc}")
            continue
        if len(fl):
            frames.append(fl)
            print(f"[flares] {p.name}: {len(fl)}")
        else:
            print(f"[flares] {p.name}: 0")
    if frames:
        cat = pd.concat(frames, ignore_index=True)
    else:
        cat = pd.DataFrame()
    if len(cat):
        cat = cat.sort_values(["time_days", "curve_file"]).reset_index(drop=True)
        cat.insert(0, "flare_id", [f"F{i:05d}" for i in range(len(cat))])
    out_dir.mkdir(parents=True, exist_ok=True)
    cat.to_csv(out_dir / "flare_catalog_v5.csv", index=False)
    return cat


# -----------------------------------------------------------------------------
# Delay targets and pair counting
# -----------------------------------------------------------------------------


def load_flic_delays(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Accept either v4 names or compatible names.
    if "tau_obs_hours" not in df.columns:
        if "tau_obs_hours_exact" in df.columns:
            df["tau_obs_hours"] = df["tau_obs_hours_exact"]
        elif "tau_obs_days" in df.columns:
            df["tau_obs_hours"] = df["tau_obs_days"] * 24.0
        else:
            raise ValueError("Delay file must contain tau_obs_hours or tau_obs_days.")
    if "target_name" not in df.columns:
        if "name" in df.columns:
            df["target_name"] = df["name"]
        else:
            df["target_name"] = [f"target_{i}" for i in range(len(df))]
    keep = []
    for _, r in df.iterrows():
        tn = str(r["target_name"])
        if "one" in tn or "two" in tn or "2over3" in tn or "4over3" in tn:
            keep.append(True)
        else:
            keep.append(False)
    out = df.loc[keep].copy()
    if len(out) == 0:
        out = df.copy()
    return out


def build_all_pairs(flares: pd.DataFrame, targets: pd.DataFrame, tolerance_hours: float, cross_band_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(flares) < 2:
        return pd.DataFrame(), pd.DataFrame()
    rows_all = []
    rows_hit = []
    f = flares.reset_index(drop=True)
    for i in range(len(f)):
        ri = f.iloc[i]
        for j in range(i + 1, len(f)):
            rj = f.iloc[j]
            if cross_band_only and ri["curve_file"] == rj["curve_file"]:
                continue
            dt_h = abs(float(rj["time_days"]) - float(ri["time_days"])) * 24.0
            base = {
                "flare_early_id": ri["flare_id"] if ri["time_days"] <= rj["time_days"] else rj["flare_id"],
                "flare_late_id": rj["flare_id"] if ri["time_days"] <= rj["time_days"] else ri["flare_id"],
                "band_early": ri["curve_id"] if ri["time_days"] <= rj["time_days"] else rj["curve_id"],
                "band_late": rj["curve_id"] if ri["time_days"] <= rj["time_days"] else ri["curve_id"],
                "file_early": ri["curve_file"] if ri["time_days"] <= rj["time_days"] else rj["curve_file"],
                "file_late": rj["curve_file"] if ri["time_days"] <= rj["time_days"] else ri["curve_file"],
                "wavelength_early_A": ri["wavelength_A"] if ri["time_days"] <= rj["time_days"] else rj["wavelength_A"],
                "wavelength_late_A": rj["wavelength_A"] if ri["time_days"] <= rj["time_days"] else ri["wavelength_A"],
                "time_early_days": float(min(ri["time_days"], rj["time_days"])),
                "time_late_days": float(max(ri["time_days"], rj["time_days"])),
                "delta_hours": float(dt_h),
                "z_early": float(ri["z_score"] if ri["time_days"] <= rj["time_days"] else rj["z_score"]),
                "z_late": float(rj["z_score"] if ri["time_days"] <= rj["time_days"] else ri["z_score"]),
                "min_abs_z": float(min(abs(ri["z_score"]), abs(rj["z_score"]))),
                "mean_abs_z": float(0.5 * (abs(ri["z_score"]) + abs(rj["z_score"]))),
                "pair_strength": float(math.sqrt(max(0.0, abs(ri["z_score"]) * abs(rj["z_score"])))),
                "same_curve": bool(ri["curve_file"] == rj["curve_file"]),
                "same_curve_id": bool(ri["curve_id"] == rj["curve_id"]),
            }
            rows_all.append(base)
            for _, target in targets.iterrows():
                tau = float(target["tau_obs_hours"])
                diff = abs(dt_h - tau)
                if diff <= tolerance_hours:
                    hit = dict(base)
                    hit.update({
                        "target_name": str(target["target_name"]),
                        "branch_label": str(target.get("branch_label", target["target_name"])),
                        "tau_flic_hours": tau,
                        "delta_minus_flic_hours": float(dt_h - tau),
                        "abs_delta_minus_flic_hours": float(diff),
                        "lambda_pair": float(dt_h / tau) if tau > 0 else np.nan,
                        "tolerance_hours": float(tolerance_hours),
                    })
                    rows_hit.append(hit)
    return pd.DataFrame(rows_all), pd.DataFrame(rows_hit)


def count_hits_for_target(all_pairs: pd.DataFrame, tau_h: float, tolerance_h: float) -> int:
    if len(all_pairs) == 0:
        return 0
    dt = all_pairs["delta_hours"].to_numpy(float)
    return int(np.sum(np.abs(dt - tau_h) <= tolerance_h))


def strength_sum_for_target(all_pairs: pd.DataFrame, tau_h: float, tolerance_h: float) -> float:
    if len(all_pairs) == 0:
        return 0.0
    m = np.abs(all_pairs["delta_hours"].to_numpy(float) - tau_h) <= tolerance_h
    if not np.any(m):
        return 0.0
    return float(np.nansum(all_pairs.loc[m, "pair_strength"].to_numpy(float)))


def summarize_hits(all_pairs: pd.DataFrame, hits: pd.DataFrame, targets: pd.DataFrame, tolerance_h: float) -> pd.DataFrame:
    rows = []
    for _, target in targets.iterrows():
        tau = float(target["tau_obs_hours"])
        tn = str(target["target_name"])
        sub = hits[hits["target_name"] == tn] if len(hits) else pd.DataFrame()
        rows.append({
            "target_name": tn,
            "branch_label": str(target.get("branch_label", tn)),
            "tau_flic_hours": tau,
            "tolerance_hours": tolerance_h,
            "n_flic_pairs": int(len(sub)),
            "sum_pair_strength": float(sub["pair_strength"].sum()) if len(sub) else 0.0,
            "median_abs_delta_hours": float(sub["abs_delta_minus_flic_hours"].median()) if len(sub) else np.nan,
            "best_abs_delta_hours": float(sub["abs_delta_minus_flic_hours"].min()) if len(sub) else np.nan,
            "strongest_pair_strength": float(sub["pair_strength"].max()) if len(sub) else np.nan,
            "n_total_pairs": int(len(all_pairs)),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Null tests
# -----------------------------------------------------------------------------


def empirical_p(null_values: Iterable[float], real_value: float) -> float:
    arr = np.asarray(list(null_values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0 or not np.isfinite(real_value):
        return np.nan
    return float((1 + np.sum(arr >= real_value)) / (len(arr) + 1))


def run_false_lag_null(all_pairs: pd.DataFrame, targets: pd.DataFrame, tolerance_h: float, n_null: int, lag_min_h: float, lag_max_h: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    # Exclude random lags close to any real target.
    target_taus = targets["tau_obs_hours"].to_numpy(float)
    for _, target in targets.iterrows():
        tn = str(target["target_name"])
        real_tau = float(target["tau_obs_hours"])
        k = 0
        tries = 0
        while k < n_null and tries < n_null * 50:
            tries += 1
            tau = float(rng.uniform(lag_min_h, lag_max_h))
            if np.any(np.abs(target_taus - tau) <= 2.0 * tolerance_h):
                continue
            rows.append({
                "null_mode": "false_lag",
                "target_name": tn,
                "null_index": k,
                "tau_test_hours": tau,
                "count_hits": count_hits_for_target(all_pairs, tau, tolerance_h),
                "sum_pair_strength": strength_sum_for_target(all_pairs, tau, tolerance_h),
            })
            k += 1
    return pd.DataFrame(rows)


def randomize_flare_times(flares: pd.DataFrame, rng: np.random.Generator, mode: str) -> pd.DataFrame:
    out = flares.copy()
    t = out["time_days"].to_numpy(float)
    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))
    span = max(1e-6, tmax - tmin)
    if mode == "uniform_time":
        out["time_days"] = rng.uniform(tmin, tmax, size=len(out))
    elif mode == "band_shift":
        new_times = t.copy()
        for curve_file, idx in out.groupby("curve_file").groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            shift = float(rng.uniform(0, span))
            new_times[idx_arr] = tmin + np.mod((t[idx_arr] - tmin) + shift, span)
        out["time_days"] = new_times
    else:
        raise ValueError(mode)
    out = out.sort_values(["time_days", "curve_file"]).reset_index(drop=True)
    return out


def _time_null_worker(
    flares: pd.DataFrame,
    targets: pd.DataFrame,
    tolerance_h: float,
    cross_band_only: bool,
    mode: str,
    n_trials: int,
    seed: int,
    null_index_offset: int,
) -> pd.DataFrame:
    """Worker for parallel time/band-shift nulls."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_trials):
        fnull = randomize_flare_times(flares, rng, mode)
        pairs_null, _ = build_all_pairs(fnull, targets, tolerance_h, cross_band_only)
        null_index = null_index_offset + k
        for _, target in targets.iterrows():
            tau = float(target["tau_obs_hours"])
            rows.append({
                "null_mode": mode,
                "target_name": str(target["target_name"]),
                "null_index": null_index,
                "tau_test_hours": tau,
                "count_hits": count_hits_for_target(pairs_null, tau, tolerance_h),
                "sum_pair_strength": strength_sum_for_target(pairs_null, tau, tolerance_h),
            })
    return pd.DataFrame(rows)


def run_time_nulls(
    flares: pd.DataFrame,
    targets: pd.DataFrame,
    tolerance_h: float,
    cross_band_only: bool,
    n_null: int,
    seed: int,
    modes: list[str],
    workers: int = 1,
) -> pd.DataFrame:
    """Run time/band-shift nulls, sequentially or in parallel."""
    workers = max(1, int(workers))
    if n_null <= 0:
        return pd.DataFrame()

    if workers == 1:
        rng = np.random.default_rng(seed)
        rows = []
        for mode in modes:
            for k in range(n_null):
                fnull = randomize_flare_times(flares, rng, mode)
                pairs_null, _ = build_all_pairs(fnull, targets, tolerance_h, cross_band_only)
                for _, target in targets.iterrows():
                    tau = float(target["tau_obs_hours"])
                    rows.append({
                        "null_mode": mode,
                        "target_name": str(target["target_name"]),
                        "null_index": k,
                        "tau_test_hours": tau,
                        "count_hits": count_hits_for_target(pairs_null, tau, tolerance_h),
                        "sum_pair_strength": strength_sum_for_target(pairs_null, tau, tolerance_h),
                    })
                if (k + 1) % 100 == 0:
                    print(f"[null {mode}] {k+1}/{n_null}")
        return pd.DataFrame(rows)

    max_workers = min(workers, os.cpu_count() or workers)
    chunk_count = min(max_workers, n_null)
    base = n_null // chunk_count
    rem = n_null % chunk_count
    tasks = []
    for mode_i, mode in enumerate(modes):
        start_idx = 0
        for c in range(chunk_count):
            n = base + (1 if c < rem else 0)
            if n <= 0:
                continue
            task_seed = seed + 100000 * mode_i + 1000 * c + 17
            tasks.append((mode, n, task_seed, start_idx))
            start_idx += n

    print(f"[parallel] time nulls: modes={modes}, n_null={n_null}, workers={max_workers}, tasks={len(tasks)}")
    frames = []
    done_trials = {mode: 0 for mode in modes}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_time_null_worker, flares, targets, tolerance_h, cross_band_only, mode, n, task_seed, offset)
            for mode, n, task_seed, offset in tasks
        ]
        for fut in as_completed(futs):
            df = fut.result()
            frames.append(df)
            if len(df):
                mode = str(df["null_mode"].iloc[0])
                n_targets = max(1, len(targets))
                done_trials[mode] += int(len(df) / n_targets)
                print(f"[null {mode}] {done_trials[mode]}/{n_null}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_pvalues(summary: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    p_count_cols = []
    p_strength_cols = []
    for mode in sorted(nulls["null_mode"].unique()) if len(nulls) else []:
        p_count = []
        p_strength = []
        for _, row in out.iterrows():
            sub = nulls[(nulls["null_mode"] == mode) & (nulls["target_name"] == row["target_name"])]
            p_count.append(empirical_p(sub["count_hits"].to_numpy(float), float(row["n_flic_pairs"])))
            p_strength.append(empirical_p(sub["sum_pair_strength"].to_numpy(float), float(row["sum_pair_strength"])))
        c1 = f"p_count_{mode}"
        c2 = f"p_strength_{mode}"
        out[c1] = p_count
        out[c2] = p_strength
        p_count_cols.append(c1)
        p_strength_cols.append(c2)
    if p_count_cols:
        out["p_count_worst"] = out[p_count_cols].max(axis=1)
        out["p_count_best"] = out[p_count_cols].min(axis=1)
    if p_strength_cols:
        out["p_strength_worst"] = out[p_strength_cols].max(axis=1)
        out["p_strength_best"] = out[p_strength_cols].min(axis=1)
    return out


# -----------------------------------------------------------------------------
# Plots and report
# -----------------------------------------------------------------------------


def plot_lag_histogram(all_pairs: pd.DataFrame, targets: pd.DataFrame, tolerance_h: float, out_path: Path) -> None:
    if plt is None or len(all_pairs) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_pairs["delta_hours"].to_numpy(float), bins=120, range=(0, 60), alpha=0.8)
    for _, target in targets.iterrows():
        tau = float(target["tau_obs_hours"])
        ax.axvline(tau, linestyle="--", linewidth=2, label=str(target["target_name"]))
        ax.axvspan(tau - tolerance_h, tau + tolerance_h, alpha=0.12)
    ax.set_xlabel("flare-pair separation, hours")
    ax.set_ylabel("number of flare pairs")
    ax.set_title("All-band flare-pair lag histogram")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_timeline(flares: pd.DataFrame, hits: pd.DataFrame, out_path: Path) -> None:
    if plt is None or len(flares) == 0:
        return
    order = {name: i for i, name in enumerate(sorted(flares["curve_id"].unique()))}
    fig, ax = plt.subplots(figsize=(12, max(5, 0.3 * len(order))))
    y = [order[x] for x in flares["curve_id"]]
    sizes = 10 + 20 * np.clip(flares["abs_z_score"].to_numpy(float), 0, 6)
    ax.scatter(flares["time_days"], y, s=sizes, alpha=0.7)
    # Draw a limited number of FLiC hit links to avoid unreadable plots.
    if len(hits):
        fdict = flares.set_index("flare_id").to_dict(orient="index")
        top = hits.sort_values("pair_strength", ascending=False).head(80)
        for _, r in top.iterrows():
            a = fdict.get(r["flare_early_id"])
            b = fdict.get(r["flare_late_id"])
            if a is None or b is None:
                continue
            ax.plot([a["time_days"], b["time_days"]], [order[a["curve_id"]], order[b["curve_id"]]], alpha=0.25, linewidth=0.8)
    ax.set_yticks(list(order.values()))
    ax.set_yticklabels(list(order.keys()))
    ax.set_xlabel("time, HJD - 2400000")
    ax.set_title("Detected all-band flare markers and strongest FLiC-pair links")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_report(out_dir: Path, flares: pd.DataFrame, pairs: pd.DataFrame, hits: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = []
    lines.append("# NGC 5548 all-band flare-pair FLiC search v5.3")
    lines.append("")
    lines.append("This is an event-based accretion test. It does not assume that one photometric band is the physical driver of another. Every significant flare marker in every selected light curve is treated as a possible accretion event marker, and all flare pairs are tested against the fixed FLiC delays.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Curves directory: `{args.curves_dir}`")
    lines.append(f"- Delay file argument: `{args.delays}`; if not found, built-in NGC 5548 targets are used.")
    lines.append(f"- Minimum flare strength: `{args.min_z}` robust sigma")
    lines.append(f"- Trend window: `{args.trend_window_days}` days")
    lines.append(f"- Minimum flare separation within one curve: `{args.min_separation_hours}` hours")
    lines.append(f"- FLiC-pair tolerance window: `±{args.tolerance_hours}` hours")
    lines.append(f"- Cross-band only: `{args.cross_band_only}`")
    lines.append(f"- Unique physical bands: `{args.unique_physical_bands}`")
    lines.append(f"- Exclude patterns: `{args.exclude}`")
    lines.append(f"- Exclude file: `{args.exclude_file}`")
    lines.append(f"- Null trials: `{args.n_null}`")
    lines.append(f"- Null worker processes: `{args.workers}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Detected flare markers: `{len(flares)}`")
    lines.append(f"- All tested flare pairs: `{len(pairs)}`")
    lines.append(f"- FLiC-window hit pairs: `{len(hits)}`")
    lines.append("")
    lines.append("## FLiC pair summary")
    lines.append("")
    if len(summary):
        cols = [c for c in ["target_name", "tau_flic_hours", "n_flic_pairs", "sum_pair_strength", "median_abs_delta_hours", "best_abs_delta_hours", "p_count_false_lag", "p_strength_false_lag", "p_count_uniform_time", "p_strength_uniform_time", "p_count_band_shift", "p_strength_band_shift"] if c in summary.columns]
        lines.append(summary[cols].to_markdown(index=False))
    else:
        lines.append("No summary rows.")
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("The key statistic is not a correlation of one chosen band against another. It is the over-density of flare pairs whose time separations fall near the fixed one-way and two-way FLiC delays. A small false-lag p-value means that random horizontal lag scales rarely collect as many flare pairs as the FLiC scale. Random-time and band-shift nulls are stricter checks against cadence and sampling artefacts.")
    lines.append("")
    lines.append("This script provides the event-based test requested for the accretion interpretation. The previous DCF scan remains useful as a correlation map, but this v5 analysis removes the privileged single-reference-band assumption.")
    (out_dir / "flare_pair_report_v5.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="All-band flare-pair FLiC search for NGC 5548")
    p.add_argument("--curves-dir", default="run_v4/curves")
    p.add_argument("--delays", default=None, help="optional delay CSV; if omitted or missing, built-in NGC 5548 FLiC delays are computed")
    p.add_argument("--out-dir", default="run_v5/results")
    p.add_argument("--curve-glob", default="*.csv")
    p.add_argument("--exclude", action="append", default=[], help="curve file/stem/canonical-id glob to exclude; may be repeated, e.g. --exclude hst_cos_1367_paper1.csv")
    p.add_argument("--exclude-file", default=None, help="optional text file with one exclude pattern per line")
    p.add_argument("--unique-physical-bands", action="store_true", help="keep only one file per canonical physical band; skips duplicates such as hst_cos_1367_paper1.csv when hst_cos_1367.csv is present")
    p.add_argument("--include-lines", action="store_true", help="include broad-line light curves; default is continuum only")
    p.add_argument("--include-xray", action="store_true", help="include Swift/XRT curves; default is off")
    p.add_argument("--min-z", type=float, default=2.0)
    p.add_argument("--trend-window-days", type=float, default=5.0)
    p.add_argument("--min-separation-hours", type=float, default=6.0)
    p.add_argument("--max-flares-per-curve", type=int, default=50)
    p.add_argument("--include-negative", action="store_true", help="also treat negative excursions as events")
    p.add_argument("--tolerance-hours", type=float, default=1.5)
    p.add_argument("--cross-band-only", action="store_true", help="only pair flares from different curve files")
    p.add_argument("--n-null", type=int, default=5000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="parallel worker processes for time/band-shift nulls; use 1 for old sequential behavior")
    p.add_argument("--lag-min-hours", type=float, default=5.0)
    p.add_argument("--lag-max-hours", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    curves_dir = Path(args.curves_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not curves_dir.exists():
        raise FileNotFoundError(f"Curves directory not found: {curves_dir}")

    delays_path = resolve_delay_file(args.delays, curves_dir)
    if delays_path is not None:
        print(f"[delays] using {delays_path}")
        targets = load_flic_delays(delays_path)
    else:
        print("[delays] delay CSV not found; using built-in NGC 5548 defaults: M=6.5e7 Msun, z=0.017175")
        targets = default_ngc5548_flic_targets()

    targets.to_csv(out_dir / "flic_delay_targets_v5.csv", index=False)
    print("[targets]")
    print(targets[["target_name", "tau_obs_hours"]].to_string(index=False))

    print("[stage] flare detection")
    if args.unique_physical_bands:
        print("[dedup] unique physical band mode enabled")
    if args.exclude:
        print(f"[exclude] patterns: {args.exclude}")
    flares = build_flare_catalog(curves_dir, out_dir, args)
    if len(flares) == 0:
        raise RuntimeError("No flare markers found. Lower --min-z or check curves.")
    print(f"[ok] flare_catalog_v5.csv: {len(flares)} flare markers")

    print("[stage] pair building")
    pairs, hits = build_all_pairs(flares, targets, args.tolerance_hours, args.cross_band_only)
    pairs.to_csv(out_dir / "flare_pairs_all_v5.csv", index=False)
    hits.to_csv(out_dir / "flare_pairs_flic_hits_v5.csv", index=False)
    print(f"[ok] all pairs: {len(pairs)}; FLiC hit pairs: {len(hits)}")

    summary0 = summarize_hits(pairs, hits, targets, args.tolerance_hours)

    print("[stage] null tests")
    null_parts = []
    null_parts.append(run_false_lag_null(pairs, targets, args.tolerance_hours, args.n_null, args.lag_min_hours, args.lag_max_hours, args.seed))
    # Time nulls are heavier because they rebuild pairs; use same n_null for consistency.
    null_parts.append(run_time_nulls(flares, targets, args.tolerance_hours, args.cross_band_only, args.n_null, args.seed + 100, ["uniform_time", "band_shift"], workers=args.workers))
    nulls = pd.concat(null_parts, ignore_index=True) if null_parts else pd.DataFrame()
    nulls.to_csv(out_dir / "flare_pair_null_distribution_v5.csv", index=False)

    summary = attach_pvalues(summary0, nulls)
    summary.to_csv(out_dir / "flare_pair_summary_v5.csv", index=False)

    # Top hit table for quick inspection.
    if len(hits):
        top = hits.sort_values(["target_name", "abs_delta_minus_flic_hours", "pair_strength"], ascending=[True, True, False])
        top.to_csv(out_dir / "flare_pairs_flic_hits_ranked_v5.csv", index=False)

    if not args.no_plots:
        print("[stage] plots")
        plot_lag_histogram(pairs, targets, args.tolerance_hours, out_dir / "flare_pair_lag_histogram_v5.png")
        plot_timeline(flares, hits, out_dir / "flare_pair_timeline_v5.png")

    write_report(out_dir, flares, pairs, hits, summary, args)
    print("[ok] wrote report:", out_dir / "flare_pair_report_v5.md")
    print("[done]")


if __name__ == "__main__":
    main()
