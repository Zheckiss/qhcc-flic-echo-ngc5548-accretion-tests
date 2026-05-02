#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qhcc_ngc5548_flare_stack_v6.py

All-band flare-triggered stacking test for QHCC/FLiC accretion echoes in NGC 5548.

This script does NOT select a single photometric band as a physical driver.
It:
  1) loads all prepared light curves from --curves-dir;
  2) detrends and robust-normalizes each curve;
  3) detects positive flare markers in every band;
  4) clusters nearby markers into unique event anchors;
  5) stacks all-band residual light curves around those anchors;
  6) measures stacked excess at fixed FLiC delays;
  7) estimates null distributions using random anchors and false lags.

The core test is:
  after stacking many flare-triggered windows, does the mean response contain
  an excess near tau = Delta t_FLIC (one-way or two-way), compared with
  sidebands and null stacks?

Default NGC 5548 targets:
  one_way_2over3: 12.796355 h
  two_way_4over3: 25.592711 h

No scipy dependency is required.
"""

from __future__ import annotations

import argparse
import math
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Use non-interactive backend for Windows/cmd and headless runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Constants and FLiC targets
# -----------------------------

G = 6.67430e-11
C = 299792458.0
MSUN = 1.98847e30
L_PLANCK = 1.616255e-35

DEFAULT_MASS_MSUN = 6.5e7
DEFAULT_REDSHIFT = 0.017175


def flic_delay_hours(mass_msun: float, z: float, alpha: float) -> float:
    """Observer-frame FLiC delay in hours."""
    mass_kg = mass_msun * MSUN
    r_s = 2.0 * G * mass_kg / (C * C)
    tau_source_s = alpha * (r_s / C) * math.log(r_s / L_PLANCK)
    tau_obs_s = (1.0 + z) * tau_source_s
    return tau_obs_s / 3600.0


def default_targets() -> pd.DataFrame:
    rows = [
        {
            "target_name": "one_way_2over3",
            "alpha": 2.0 / 3.0,
            "tau_obs_hours": flic_delay_hours(DEFAULT_MASS_MSUN, DEFAULT_REDSHIFT, 2.0 / 3.0),
        },
        {
            "target_name": "two_way_4over3",
            "alpha": 4.0 / 3.0,
            "tau_obs_hours": flic_delay_hours(DEFAULT_MASS_MSUN, DEFAULT_REDSHIFT, 4.0 / 3.0),
        },
    ]
    return pd.DataFrame(rows)


def load_targets(delay_csv: Optional[Path]) -> pd.DataFrame:
    if delay_csv is None or not delay_csv.exists():
        return default_targets()

    df = pd.read_csv(delay_csv)
    # Flexible mapping from earlier delay-table variants.
    name_col = None
    tau_col = None
    alpha_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("target_name", "branch", "name"):
            name_col = c
        if cl in ("tau_obs_hours", "delay_obs_hours", "tau_hours", "delay_hours"):
            tau_col = c
        if cl == "alpha":
            alpha_col = c

    # If only days are available, convert.
    if tau_col is None:
        for c in df.columns:
            cl = c.lower()
            if cl in ("tau_obs_days", "delay_obs_days", "tau_days", "delay_days"):
                df["tau_obs_hours"] = pd.to_numeric(df[c], errors="coerce") * 24.0
                tau_col = "tau_obs_hours"
                break

    if tau_col is None:
        return default_targets()

    out = pd.DataFrame()
    out["tau_obs_hours"] = pd.to_numeric(df[tau_col], errors="coerce")
    if name_col is not None:
        out["target_name"] = df[name_col].astype(str)
    else:
        out["target_name"] = [f"target_{i}" for i in range(len(out))]
    if alpha_col is not None:
        out["alpha"] = pd.to_numeric(df[alpha_col], errors="coerce")
    else:
        out["alpha"] = np.nan

    out = out[np.isfinite(out["tau_obs_hours"])].copy()
    # Keep main 2/3 and 4/3 targets if names are messy.
    keep = out["target_name"].str.contains("one|two|2over3|4over3|2/3|4/3", case=False, regex=True)
    if keep.any():
        out = out[keep].copy()
    if len(out) == 0:
        return default_targets()
    return out.reset_index(drop=True)


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class CurveData:
    file: str
    physical_band: str
    wavelength_A: float
    t_days: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    trend: np.ndarray
    residual: np.ndarray
    z: np.ndarray


@dataclass
class FlareMarker:
    flare_id: int
    file: str
    physical_band: str
    wavelength_A: float
    t_days: float
    z_peak: float
    residual: float
    flux: float


@dataclass
class AnchorEvent:
    anchor_id: int
    t_anchor_days: float
    max_z: float
    n_markers: int
    bands: str
    files: str
    marker_ids: str


# -----------------------------
# Helpers
# -----------------------------

def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.nanstd(x))
    if not np.isfinite(sig) or sig <= 0:
        sig = 1.0
    return sig


def physical_band_from_name(path: Path) -> str:
    stem = path.stem
    s = stem.lower()
    s = s.replace("_paper1", "")
    s = s.replace("_daily", "")
    # Normalize common names.
    if s.startswith("hst_cos_"):
        m = re.search(r"hst_cos_(\d+)", s)
        if m:
            return f"hst_cos_{m.group(1)}"
    if s.startswith("swift_"):
        return s
    if s.startswith("opt_"):
        return s
    return s


def wavelength_from_band(physical_band: str) -> float:
    # Central/effective wavelengths in Angstroms, approximate.
    # These are used only for labels/plots, not for the core timing test.
    mapping = {
        "hst_cos_1158": 1158.0,
        "hst_cos_1367": 1367.0,
        "hst_cos_1478": 1478.0,
        "hst_cos_1746": 1746.0,
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
    return mapping.get(physical_band.lower(), np.nan)


def choose_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    time_candidates = [
        "t", "time", "time_days", "hjd", "mjd", "jd", "hjd_minus_2400000",
        "hjd_2400000", "date", "day"
    ]
    flux_candidates = [
        "flux", "f", "value", "rate", "count_rate", "flux_density",
        "continuum", "mag", "magnitude"
    ]
    err_candidates = [
        "flux_err", "err", "error", "e_flux", "sigma", "unc", "uncertainty",
        "mag_err", "e_mag"
    ]

    t_col = None
    for k in time_candidates:
        if k in lower:
            t_col = lower[k]
            break
    if t_col is None:
        # Pick first mostly numeric column as time.
        numeric = []
        for c in cols:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() >= max(3, int(0.5 * len(df))):
                numeric.append(c)
        if not numeric:
            raise ValueError("Could not find numeric time column")
        t_col = numeric[0]

    flux_col = None
    for k in flux_candidates:
        if k in lower and lower[k] != t_col:
            flux_col = lower[k]
            break
    if flux_col is None:
        numeric = []
        for c in cols:
            if c == t_col:
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() >= max(3, int(0.5 * len(df))):
                numeric.append(c)
        if not numeric:
            raise ValueError("Could not find numeric flux column")
        flux_col = numeric[0]

    err_col = None
    for k in err_candidates:
        if k in lower and lower[k] not in (t_col, flux_col):
            err_col = lower[k]
            break
    return t_col, flux_col, err_col


def normalize_time_days(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    # Keep relative times consistent. Prepared curves are normally HJD-2400000.
    # If absolute JD/HJD is present, shift to HJD-2400000-like scale.
    med = np.nanmedian(t)
    if med > 2_400_000:
        t = t - 2_400_000.0
    elif med > 50_000 and med < 2_400_000:
        # MJD-like or HJD-2400000-like; leave as is.
        t = t
    return t


def local_median_trend(t: np.ndarray, y: np.ndarray, window_days: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    trend = np.empty_like(y, dtype=float)
    half = window_days / 2.0
    for i, ti in enumerate(t):
        m = np.isfinite(y) & (np.abs(t - ti) <= half)
        if m.sum() >= 3:
            trend[i] = np.nanmedian(y[m])
        else:
            trend[i] = np.nanmedian(y[np.isfinite(y)])
    return trend


def load_curve(path: Path, trend_window_days: float) -> CurveData:
    df = pd.read_csv(path)
    t_col, f_col, e_col = choose_columns(df)
    t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    f = pd.to_numeric(df[f_col], errors="coerce").to_numpy(dtype=float)
    if e_col is not None:
        e = pd.to_numeric(df[e_col], errors="coerce").to_numpy(dtype=float)
    else:
        e = np.full_like(f, np.nan, dtype=float)

    m = np.isfinite(t) & np.isfinite(f)
    t = normalize_time_days(t[m])
    f = f[m]
    e = e[m] if len(e) == len(m) else np.full_like(f, np.nan)

    order = np.argsort(t)
    t, f, e = t[order], f[order], e[order]

    # If curve is in magnitudes, invert sign so positive z means brightening.
    # Heuristic: filenames with opt/swift generally already prepared as flux.
    # Only invert when flux column says mag.
    if "mag" in f_col.lower():
        f = -f

    trend = local_median_trend(t, f, trend_window_days)
    resid = f - trend
    sig = robust_sigma(resid)
    z = resid / sig

    phys = physical_band_from_name(path)
    wave = wavelength_from_band(phys)
    return CurveData(
        file=path.name,
        physical_band=phys,
        wavelength_A=wave,
        t_days=t,
        flux=f,
        flux_err=e,
        trend=trend,
        residual=resid,
        z=z,
    )


def load_all_curves(
    curves_dir: Path,
    trend_window_days: float,
    unique_physical_bands: bool,
    exclude_patterns: List[str],
) -> Tuple[List[CurveData], pd.DataFrame]:
    files = sorted(curves_dir.glob("*.csv"))
    inventory_rows = []
    selected: List[CurveData] = []
    seen_bands: set = set()

    for path in files:
        skip_reason = ""
        if any(re.search(pat, path.name, flags=re.IGNORECASE) for pat in exclude_patterns):
            skip_reason = "excluded_by_pattern"

        phys = physical_band_from_name(path)
        if not skip_reason and unique_physical_bands and phys in seen_bands:
            skip_reason = f"duplicate_physical_band:{phys}"

        if skip_reason:
            inventory_rows.append({
                "file": path.name,
                "physical_band": phys,
                "selected": False,
                "skip_reason": skip_reason,
                "n": np.nan,
                "t_min": np.nan,
                "t_max": np.nan,
            })
            continue

        try:
            c = load_curve(path, trend_window_days)
            selected.append(c)
            seen_bands.add(c.physical_band)
            inventory_rows.append({
                "file": path.name,
                "physical_band": c.physical_band,
                "selected": True,
                "skip_reason": "",
                "n": len(c.t_days),
                "t_min": float(np.nanmin(c.t_days)),
                "t_max": float(np.nanmax(c.t_days)),
                "wavelength_A": c.wavelength_A,
            })
        except Exception as exc:
            inventory_rows.append({
                "file": path.name,
                "physical_band": phys,
                "selected": False,
                "skip_reason": f"load_error:{type(exc).__name__}:{exc}",
                "n": np.nan,
                "t_min": np.nan,
                "t_max": np.nan,
            })

    return selected, pd.DataFrame(inventory_rows)


# -----------------------------
# Flare detection and anchors
# -----------------------------

def detect_flares_in_curve(
    curve: CurveData,
    min_z: float,
    min_sep_hours: float,
) -> List[FlareMarker]:
    t = curve.t_days
    z = curve.z
    markers: List[FlareMarker] = []
    if len(t) < 3:
        return markers

    candidates = []
    for i in range(1, len(t) - 1):
        if not np.isfinite(z[i]):
            continue
        if z[i] >= min_z and z[i] >= z[i - 1] and z[i] > z[i + 1]:
            candidates.append(i)

    # Include endpoints if strong and isolated.
    if len(t) >= 1 and np.isfinite(z[0]) and z[0] >= min_z and z[0] > z[1]:
        candidates.append(0)
    if len(t) >= 2 and np.isfinite(z[-1]) and z[-1] >= min_z and z[-1] >= z[-2]:
        candidates.append(len(t) - 1)

    # Greedy non-maximum suppression by z.
    candidates = sorted(set(candidates), key=lambda i: z[i], reverse=True)
    kept: List[int] = []
    min_sep_days = min_sep_hours / 24.0
    for i in candidates:
        if all(abs(t[i] - t[j]) >= min_sep_days for j in kept):
            kept.append(i)

    kept = sorted(kept, key=lambda i: t[i])
    for i in kept:
        markers.append(FlareMarker(
            flare_id=-1,
            file=curve.file,
            physical_band=curve.physical_band,
            wavelength_A=float(curve.wavelength_A) if np.isfinite(curve.wavelength_A) else np.nan,
            t_days=float(t[i]),
            z_peak=float(z[i]),
            residual=float(curve.residual[i]),
            flux=float(curve.flux[i]),
        ))
    return markers


def detect_all_flares(curves: List[CurveData], min_z: float, min_sep_hours: float) -> pd.DataFrame:
    all_markers: List[FlareMarker] = []
    next_id = 0
    for curve in curves:
        markers = detect_flares_in_curve(curve, min_z=min_z, min_sep_hours=min_sep_hours)
        for m in markers:
            m.flare_id = next_id
            next_id += 1
            all_markers.append(m)
        print(f"[flares] {curve.file}: {len(markers)}")

    if not all_markers:
        return pd.DataFrame(columns=[f.name for f in FlareMarker.__dataclass_fields__.values()])
    df = pd.DataFrame([asdict(m) for m in all_markers])
    return df.sort_values("t_days").reset_index(drop=True)


def cluster_anchors(flare_df: pd.DataFrame, cluster_hours: float) -> pd.DataFrame:
    if len(flare_df) == 0:
        return pd.DataFrame(columns=[f.name for f in AnchorEvent.__dataclass_fields__.values()])

    df = flare_df.sort_values("t_days").reset_index(drop=True)
    cluster_days = cluster_hours / 24.0

    anchors: List[AnchorEvent] = []
    current = [0]

    for i in range(1, len(df)):
        # Cluster by distance from last marker in cluster. This captures chains.
        if float(df.loc[i, "t_days"]) - float(df.loc[current[-1], "t_days"]) <= cluster_days:
            current.append(i)
        else:
            anchors.append(make_anchor(len(anchors), df.loc[current]))
            current = [i]
    if current:
        anchors.append(make_anchor(len(anchors), df.loc[current]))

    return pd.DataFrame([asdict(a) for a in anchors])


def make_anchor(anchor_id: int, sub: pd.DataFrame) -> AnchorEvent:
    weights = np.maximum(pd.to_numeric(sub["z_peak"], errors="coerce").to_numpy(dtype=float), 0.1)
    times = pd.to_numeric(sub["t_days"], errors="coerce").to_numpy(dtype=float)
    t_anchor = float(np.average(times, weights=weights))
    return AnchorEvent(
        anchor_id=anchor_id,
        t_anchor_days=t_anchor,
        max_z=float(np.nanmax(weights)),
        n_markers=int(len(sub)),
        bands=";".join(sorted(set(sub["physical_band"].astype(str)))),
        files=";".join(sorted(set(sub["file"].astype(str)))),
        marker_ids=";".join(str(int(x)) for x in sub["flare_id"].tolist()),
    )


# -----------------------------
# Stacking
# -----------------------------

def stack_profile(
    curves: List[CurveData],
    anchor_times_days: np.ndarray,
    before_hours: float,
    after_hours: float,
    bin_hours: float,
) -> pd.DataFrame:
    edges = np.arange(-before_hours, after_hours + bin_hours, bin_hours)
    if edges[-1] < after_hours:
        edges = np.append(edges, after_hours)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Per-anchor bin means, then average anchors so dense curves do not dominate.
    anchor_bin_values: List[List[float]] = [[] for _ in centers]
    point_counts = np.zeros(len(centers), dtype=int)

    before_days = before_hours / 24.0
    after_days = after_hours / 24.0

    for t0 in anchor_times_days:
        vals_by_bin: List[List[float]] = [[] for _ in centers]
        lo = t0 - before_days
        hi = t0 + after_days

        for c in curves:
            m = (c.t_days >= lo) & (c.t_days <= hi) & np.isfinite(c.z)
            if not np.any(m):
                continue
            tau_h = (c.t_days[m] - t0) * 24.0
            zvals = c.z[m]
            idx = np.searchsorted(edges, tau_h, side="right") - 1
            valid = (idx >= 0) & (idx < len(centers))
            for bi, zv in zip(idx[valid], zvals[valid]):
                vals_by_bin[int(bi)].append(float(zv))
                point_counts[int(bi)] += 1

        for bi, vals in enumerate(vals_by_bin):
            if vals:
                anchor_bin_values[bi].append(float(np.nanmean(vals)))

    rows = []
    for i, cen in enumerate(centers):
        vals = np.asarray(anchor_bin_values[i], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            mean = median = sem = np.nan
        else:
            mean = float(np.nanmean(vals))
            median = float(np.nanmedian(vals))
            sem = float(np.nanstd(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else np.nan
        rows.append({
            "tau_hours": float(cen),
            "mean_z": mean,
            "median_z": median,
            "sem_z": sem,
            "n_anchor_bins": int(len(vals)),
            "n_points_total": int(point_counts[i]),
        })
    return pd.DataFrame(rows)


def window_mean(profile: pd.DataFrame, center_h: float, half_width_h: float, col: str = "mean_z") -> Tuple[float, int]:
    m = (profile["tau_hours"] >= center_h - half_width_h) & (profile["tau_hours"] <= center_h + half_width_h)
    vals = pd.to_numeric(profile.loc[m, col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, 0
    return float(np.nanmean(vals)), int(len(vals))


def local_sideband_mean(
    profile: pd.DataFrame,
    center_h: float,
    half_width_h: float,
    sideband_factor: float = 3.0,
    col: str = "mean_z",
) -> Tuple[float, int]:
    w = half_width_h
    lo1, hi1 = center_h - sideband_factor * w, center_h - w
    lo2, hi2 = center_h + w, center_h + sideband_factor * w
    m = ((profile["tau_hours"] >= lo1) & (profile["tau_hours"] < hi1)) | ((profile["tau_hours"] > lo2) & (profile["tau_hours"] <= hi2))
    vals = pd.to_numeric(profile.loc[m, col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, 0
    return float(np.nanmean(vals)), int(len(vals))


def echo_stats_for_targets(profile: pd.DataFrame, targets: pd.DataFrame, half_width_h: float) -> pd.DataFrame:
    rows = []
    for _, tr in targets.iterrows():
        tau = float(tr["tau_obs_hours"])
        echo_mean, n_echo = window_mean(profile, tau, half_width_h)
        side_mean, n_side = local_sideband_mean(profile, tau, half_width_h)
        excess = echo_mean - side_mean if np.isfinite(echo_mean) and np.isfinite(side_mean) else np.nan
        rows.append({
            "target_name": str(tr["target_name"]),
            "tau_flic_hours": tau,
            "echo_half_width_hours": half_width_h,
            "echo_mean_z": echo_mean,
            "sideband_mean_z": side_mean,
            "echo_excess_z": excess,
            "n_echo_bins": n_echo,
            "n_sideband_bins": n_side,
        })
    return pd.DataFrame(rows)


# -----------------------------
# Nulls
# -----------------------------

def _random_anchor_worker(args):
    seed, n_trials, curve_payload, time_min, time_max, n_anchors, before_h, after_h, bin_h, targets_records, half_width_h = args
    rng = np.random.default_rng(seed)
    curves = []
    for item in curve_payload:
        curves.append(CurveData(
            file=item["file"],
            physical_band=item["physical_band"],
            wavelength_A=item["wavelength_A"],
            t_days=np.asarray(item["t_days"], dtype=float),
            flux=np.asarray(item["flux"], dtype=float),
            flux_err=np.asarray(item["flux_err"], dtype=float),
            trend=np.asarray(item["trend"], dtype=float),
            residual=np.asarray(item["residual"], dtype=float),
            z=np.asarray(item["z"], dtype=float),
        ))
    targets = pd.DataFrame(targets_records)
    rows = []
    for trial in range(n_trials):
        anchors = rng.uniform(time_min, time_max, size=n_anchors)
        prof = stack_profile(curves, anchors, before_h, after_h, bin_h)
        st = echo_stats_for_targets(prof, targets, half_width_h)
        for _, r in st.iterrows():
            rows.append({
                "null_type": "random_anchor",
                "trial": trial,
                "target_name": r["target_name"],
                "tau_test_hours": r["tau_flic_hours"],
                "echo_excess_z": r["echo_excess_z"],
                "echo_mean_z": r["echo_mean_z"],
                "sideband_mean_z": r["sideband_mean_z"],
            })
    return rows


def false_lag_null(profile: pd.DataFrame, targets: pd.DataFrame, half_width_h: float, n_null: int, lag_min_h: float, lag_max_h: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true_taus = [float(x) for x in targets["tau_obs_hours"]]
    rows = []
    # For each target, draw false lags excluding all true target windows.
    for _, tr in targets.iterrows():
        target_name = str(tr["target_name"])
        count = 0
        attempts = 0
        while count < n_null and attempts < n_null * 100:
            attempts += 1
            tau = float(rng.uniform(lag_min_h, lag_max_h))
            if any(abs(tau - tt) <= 2.0 * half_width_h for tt in true_taus):
                continue
            echo_mean, _ = window_mean(profile, tau, half_width_h)
            side_mean, _ = local_sideband_mean(profile, tau, half_width_h)
            excess = echo_mean - side_mean if np.isfinite(echo_mean) and np.isfinite(side_mean) else np.nan
            rows.append({
                "null_type": "false_lag",
                "trial": count,
                "target_name": target_name,
                "tau_test_hours": tau,
                "echo_excess_z": excess,
                "echo_mean_z": echo_mean,
                "sideband_mean_z": side_mean,
            })
            count += 1
    return pd.DataFrame(rows)


def run_random_anchor_nulls(
    curves: List[CurveData],
    targets: pd.DataFrame,
    n_null: int,
    n_anchors: int,
    before_h: float,
    after_h: float,
    bin_h: float,
    half_width_h: float,
    workers: int,
    seed: int,
) -> pd.DataFrame:
    if n_null <= 0:
        return pd.DataFrame()

    t_min = max(float(np.nanmin(c.t_days)) for c in curves)
    t_max = min(float(np.nanmax(c.t_days)) for c in curves)
    # Keep enough margin for stack windows.
    t_min += before_h / 24.0
    t_max -= after_h / 24.0
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        # Fall back to global span if overlap is too strict.
        t_min = min(float(np.nanmin(c.t_days)) for c in curves) + before_h / 24.0
        t_max = max(float(np.nanmax(c.t_days)) for c in curves) - after_h / 24.0

    curve_payload = []
    for c in curves:
        curve_payload.append({
            "file": c.file,
            "physical_band": c.physical_band,
            "wavelength_A": c.wavelength_A,
            "t_days": c.t_days.tolist(),
            "flux": c.flux.tolist(),
            "flux_err": c.flux_err.tolist(),
            "trend": c.trend.tolist(),
            "residual": c.residual.tolist(),
            "z": c.z.tolist(),
        })

    targets_records = targets.to_dict(orient="records")
    workers = max(1, int(workers))
    if workers == 1:
        return pd.DataFrame(_random_anchor_worker((seed, n_null, curve_payload, t_min, t_max, n_anchors, before_h, after_h, bin_h, targets_records, half_width_h)))

    chunks = []
    base_n = n_null // workers
    rem = n_null % workers
    for i in range(workers):
        n_i = base_n + (1 if i < rem else 0)
        if n_i <= 0:
            continue
        chunks.append((seed + 1009 * i, n_i, curve_payload, t_min, t_max, n_anchors, before_h, after_h, bin_h, targets_records, half_width_h))

    rows = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
        futures = [ex.submit(_random_anchor_worker, ch) for ch in chunks]
        for fut in as_completed(futures):
            rows.extend(fut.result())
    return pd.DataFrame(rows)


def add_null_pvalues(stats: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    out = stats.copy()
    for idx, r in out.iterrows():
        target = r["target_name"]
        real = float(r["echo_excess_z"])
        for null_type in sorted(nulls["null_type"].dropna().unique()):
            vals = pd.to_numeric(nulls[(nulls["target_name"] == target) & (nulls["null_type"] == null_type)]["echo_excess_z"], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0 or not np.isfinite(real):
                p = np.nan
            else:
                p = (1.0 + float(np.sum(vals >= real))) / (1.0 + len(vals))
            out.loc[idx, f"p_excess_{null_type}"] = p
            out.loc[idx, f"n_null_{null_type}"] = len(vals)
    return out


# -----------------------------
# Plots and report
# -----------------------------

def plot_stack(profile: pd.DataFrame, stats: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = profile["tau_hours"].to_numpy(dtype=float)
    y = profile["mean_z"].to_numpy(dtype=float)
    sem = profile["sem_z"].to_numpy(dtype=float)

    ax.plot(x, y, linewidth=1.5, label="stacked all-band residual")
    finite = np.isfinite(sem)
    ax.fill_between(x[finite], y[finite] - sem[finite], y[finite] + sem[finite], alpha=0.2, label="SEM")

    ax.axhline(0.0, linewidth=1.0)
    for _, r in stats.iterrows():
        tau = float(r["tau_flic_hours"])
        hw = float(r["echo_half_width_hours"])
        name = str(r["target_name"])
        ax.axvline(tau, linestyle="--", linewidth=1.3, label=f"{name}: {tau:.2f} h")
        ax.axvspan(tau - hw, tau + hw, alpha=0.12)

    ax.set_xlabel("Time relative to flare anchor, hours")
    ax.set_ylabel("Stacked robust-normalized residual")
    ax.set_title("NGC 5548 all-band flare-triggered stack")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_nulls(stats: pd.DataFrame, nulls: pd.DataFrame, out_path: Path) -> None:
    targets = list(stats["target_name"].astype(str))
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(9, 4.2 * max(1, n)), squeeze=False)
    for i, target in enumerate(targets):
        ax = axes[i, 0]
        real = float(stats.loc[stats["target_name"] == target, "echo_excess_z"].iloc[0])
        sub = nulls[nulls["target_name"] == target]
        for null_type in sorted(sub["null_type"].dropna().unique()):
            vals = pd.to_numeric(sub[sub["null_type"] == null_type]["echo_excess_z"], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                ax.hist(vals, bins=35, alpha=0.45, label=null_type)
        ax.axvline(real, color="black", linewidth=2.0, label=f"real {real:.3f}")
        ax.set_title(f"Null distribution for {target}")
        ax.set_xlabel("Echo-window excess")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(
    out_path: Path,
    args: argparse.Namespace,
    curves: List[CurveData],
    flare_df: pd.DataFrame,
    anchors: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# NGC 5548 all-band flare-triggered stacking test v6")
    lines.append("")
    lines.append("This is an event-based stacking test. It does not assign one band as a physical driver.")
    lines.append("Each unique flare event is used as a time anchor, all bands are stacked around those anchors,")
    lines.append("and the stacked profile is tested for excess at the fixed FLiC delay scales.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Curves directory: `{args.curves_dir}`")
    lines.append(f"- Minimum flare strength: `{args.min_z}` robust sigma")
    lines.append(f"- Unique physical bands: `{args.unique_physical_bands}`")
    lines.append(f"- Trend window: `{args.trend_window_days}` days")
    lines.append(f"- Minimum flare separation within one curve: `{args.min_separation_hours}` hours")
    lines.append(f"- Anchor cluster width: `{args.anchor_cluster_hours}` hours")
    lines.append(f"- Stack window: `-{args.window_before_hours}` to `+{args.window_after_hours}` hours")
    lines.append(f"- Stack bin width: `{args.bin_hours}` hours")
    lines.append(f"- Echo half-window: `±{args.echo_half_width_hours}` hours")
    lines.append(f"- Null trials: `{args.n_null}`")
    lines.append(f"- Null worker processes: `{args.workers}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Selected curves: `{len(curves)}`")
    lines.append(f"- Flare markers before anchor clustering: `{len(flare_df)}`")
    lines.append(f"- Unique flare anchors after clustering: `{len(anchors)}`")
    lines.append("")
    lines.append("## Echo-window statistics")
    lines.append("")
    lines.append(stats.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The key quantity is the stacked excess in a fixed FLiC window relative to local sidebands.")
    lines.append("A small random-anchor p-value means that random event times rarely produce the same stacked excess.")
    lines.append("A small false-lag p-value means that random lag locations in the same stack rarely produce an equal or larger excess.")
    lines.append("")
    lines.append("This v6 test is not a DCF pair-count test. It asks whether a coherent all-band response survives stacking around flare anchors.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--curves-dir", required=True)
    p.add_argument("--out-dir", default="run_v6/results")
    p.add_argument("--delays", default=None, help="Optional delay CSV. If omitted/not found, built-in NGC 5548 targets are used.")

    p.add_argument("--min-z", type=float, default=2.0)
    p.add_argument("--trend-window-days", type=float, default=5.0)
    p.add_argument("--min-separation-hours", type=float, default=6.0)
    p.add_argument("--anchor-cluster-hours", type=float, default=6.0)

    p.add_argument("--window-before-hours", type=float, default=10.0)
    p.add_argument("--window-after-hours", type=float, default=80.0)
    p.add_argument("--bin-hours", type=float, default=1.0)
    p.add_argument("--echo-half-width-hours", type=float, default=1.5)

    p.add_argument("--n-null", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=12345)

    p.add_argument("--unique-physical-bands", action="store_true")
    p.add_argument("--exclude", action="append", default=[], help="Regex filename pattern to exclude; may be repeated.")

    p.add_argument("--false-lag-min-hours", type=float, default=5.0)
    p.add_argument("--false-lag-max-hours", type=float, default=40.0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    curves_dir = Path(args.curves_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not curves_dir.exists():
        raise FileNotFoundError(f"Curves directory not found: {curves_dir}")

    delay_path = Path(args.delays) if args.delays else None
    targets = load_targets(delay_path)
    targets.to_csv(out_dir / "flic_delay_targets_v6.csv", index=False)

    print("[targets]")
    print(targets[["target_name", "tau_obs_hours"]].to_string(index=False))

    print("[stage] loading curves")
    curves, inventory = load_all_curves(
        curves_dir=curves_dir,
        trend_window_days=args.trend_window_days,
        unique_physical_bands=args.unique_physical_bands,
        exclude_patterns=args.exclude,
    )
    inventory.to_csv(out_dir / "curve_inventory_v6.csv", index=False)
    print(f"[curves] selected {len(curves)} curves")
    if len(curves) == 0:
        raise RuntimeError("No curves loaded")

    print("[stage] flare detection")
    flare_df = detect_all_flares(curves, min_z=args.min_z, min_sep_hours=args.min_separation_hours)
    flare_df.to_csv(out_dir / "flare_marker_catalog_v6.csv", index=False)
    print(f"[ok] flare_marker_catalog_v6.csv: {len(flare_df)} markers")

    print("[stage] anchor clustering")
    anchors = cluster_anchors(flare_df, cluster_hours=args.anchor_cluster_hours)
    anchors.to_csv(out_dir / "anchor_catalog_v6.csv", index=False)
    print(f"[ok] anchor_catalog_v6.csv: {len(anchors)} unique anchors")

    if len(anchors) == 0:
        raise RuntimeError("No anchors found. Lower --min-z or check curves.")

    print("[stage] stacking")
    anchor_times = anchors["t_anchor_days"].to_numpy(dtype=float)
    profile = stack_profile(
        curves,
        anchor_times,
        before_hours=args.window_before_hours,
        after_hours=args.window_after_hours,
        bin_hours=args.bin_hours,
    )
    profile.to_csv(out_dir / "stack_profile_allbands_v6.csv", index=False)

    stats = echo_stats_for_targets(profile, targets, args.echo_half_width_hours)
    print("[stage] nulls: random anchors")
    null_random = run_random_anchor_nulls(
        curves=curves,
        targets=targets,
        n_null=args.n_null,
        n_anchors=len(anchors),
        before_h=args.window_before_hours,
        after_h=args.window_after_hours,
        bin_h=args.bin_hours,
        half_width_h=args.echo_half_width_hours,
        workers=args.workers,
        seed=args.seed,
    )

    print("[stage] nulls: false lags")
    null_false = false_lag_null(
        profile=profile,
        targets=targets,
        half_width_h=args.echo_half_width_hours,
        n_null=args.n_null,
        lag_min_h=args.false_lag_min_hours,
        lag_max_h=args.false_lag_max_hours,
        seed=args.seed + 77,
    )

    nulls = pd.concat([null_random, null_false], ignore_index=True)
    nulls.to_csv(out_dir / "stack_null_distribution_v6.csv", index=False)

    stats = add_null_pvalues(stats, nulls)
    stats.to_csv(out_dir / "echo_window_stats_v6.csv", index=False)

    print("[stage] plots")
    plot_stack(profile, stats, out_dir / "flare_stack_profile_v6.png")
    plot_nulls(stats, nulls, out_dir / "flare_stack_nulls_v6.png")

    write_report(out_dir / "flare_stack_report_v6.md", args, curves, flare_df, anchors, stats)

    print("[done]")
    print(out_dir / "echo_window_stats_v6.csv")
    print(out_dir / "flare_stack_report_v6.md")


if __name__ == "__main__":
    main()
