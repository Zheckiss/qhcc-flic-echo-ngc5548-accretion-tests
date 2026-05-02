#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qhcc_ngc5548_core_v3.py

Чистое ядро пайплайна QHCC/FLiC для NGC 5548.

Главные исправления v3:
1. Временная шкала нормализуется автоматически:
   - полный HJD, например 2456875 -> HJD-2400000 = 56875;
   - HJD-2450000, например 6690 -> HJD-2400000 = 56690;
   - HJD-2400000, например 56690 -> без изменений.

2. Корреляционный score называется DCF-score, а не Pearson correlation.

3. В итогах есть две статистики:
   - target: значение строго около заранее предсказанного лага;
   - peak: максимум внутри заранее заданного окна.

4. Нулевые тесты считаются внутри одного процесса, без тысяч запусков Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import requests
import numpy as np
import pandas as pd


# ---------- Константы ----------

G = 6.67430e-11
C = 299_792_458.0
M_SUN = 1.98847e30
L_PLANCK = 1.616255e-35
DAY = 86400.0

TAU_PRIMARY = 0.599829156843
TAU_DOUBLE = 1.199658313686

RAW_FILES = {
    "J_ApJ_806_128_table2.dat": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/806/128/table2.dat",
    "J_ApJ_806_129_table2.dat": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/806/129/table2.dat",
    "J_ApJ_806_129_table3.dat": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/806/129/table3.dat",
    "J_ApJ_821_56_table3.dat": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/821/56/table3.dat",
    "J_ApJ_821_56_table4.dat": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/821/56/table4.dat",
}


# ---------- Общие функции ----------

def ensure_dirs(work_dir: Path) -> dict[str, Path]:
    raw_dir = work_dir / "raw"
    curves_dir = work_dir / "curves"
    results_dir = work_dir / "results"
    for p in [raw_dir, curves_dir, results_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "curves": curves_dir, "results": results_dir}


def normalize_hjd_to_2400000(series: pd.Series) -> pd.Series:
    """
    Приводит разные форматы времени к HJD - 2400000.

    Возможные исходные шкалы:
    - полный HJD:       2456875 -> 56875
    - HJD - 2450000:      6690 -> 56690
    - HJD - 2400000:     56690 -> 56690
    """
    s = pd.to_numeric(series, errors="coerce")
    med = float(np.nanmedian(s))

    if not np.isfinite(med):
        return s

    if med > 1_000_000:
        return s - 2_400_000.0

    if med < 20_000:
        return s + 50_000.0

    return s


def save_curve(df: pd.DataFrame, out_path: Path) -> dict:
    out = df[["time", "flux", "flux_err"]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.sort_values("time")
    out.to_csv(out_path, index=False)

    if len(out) == 0:
        return {"file": out_path.name, "n": 0, "t_min": np.nan, "t_max": np.nan, "t_median": np.nan}

    return {
        "file": out_path.name,
        "n": int(len(out)),
        "t_min": float(out["time"].min()),
        "t_max": float(out["time"].max()),
        "t_median": float(out["time"].median()),
    }


def load_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"time", "flux", "flux_err"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: нет колонок {sorted(missing)}")
    df = df[["time", "flux", "flux_err"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.sort_values("time").reset_index(drop=True)


def write_manifest(curves_dir: Path, out_csv: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(curves_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p)
            t = pd.to_numeric(df["time"], errors="coerce").dropna()
            rows.append({
                "file": p.name,
                "n": int(len(t)),
                "t_min": float(t.min()) if len(t) else np.nan,
                "t_max": float(t.max()) if len(t) else np.nan,
                "t_median": float(t.median()) if len(t) else np.nan,
            })
        except Exception as exc:
            rows.append({"file": p.name, "n": 0, "t_min": np.nan, "t_max": np.nan, "t_median": np.nan, "error": str(exc)})

    man = pd.DataFrame(rows)
    man.to_csv(out_csv, index=False)
    return man


# ---------- Загрузка и подготовка данных ----------

def download_all(raw_dir: Path, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    for name, url in RAW_FILES.items():
        out = raw_dir / name

        if out.exists() and out.stat().st_size > 0 and not force:
            print(f"[ok] уже есть: {out}")
            continue

        print(f"[download] {url}")
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "qhcc-flic-ngc5548-v3"})
            r.raise_for_status()
        except Exception as exc:
            failures.append((name, url, exc))
            print(f"[fail] {name}: {exc}")
            continue

        out.write_bytes(r.content)
        print(f"[ok] записан: {out} ({out.stat().st_size} bytes)")

    if failures:
        print("")
        print("[STOP] Не все файлы скачались.")
        print("Скачай вручную и положи в raw с такими именами:")
        for name, url, exc in failures:
            print(f"  {name}")
            print(f"  {url}")
        raise SystemExit(2)


def weighted_bin_average(df: pd.DataFrame, bin_days: float = 0.25) -> pd.DataFrame:
    x = df[["time", "flux", "flux_err"]].dropna().copy()
    if len(x) == 0:
        return x

    x["bin"] = np.round(x["time"] / bin_days).astype(int)
    rows = []

    for _, g in x.groupby("bin"):
        err = g["flux_err"].to_numpy(float)
        flux = g["flux"].to_numpy(float)
        time = g["time"].to_numpy(float)

        good = np.isfinite(err) & (err > 0) & np.isfinite(flux) & np.isfinite(time)
        if good.sum() == 0:
            continue

        w = 1.0 / err[good] ** 2
        rows.append({
            "time": float(np.mean(time[good])),
            "flux": float(np.sum(w * flux[good]) / np.sum(w)),
            "flux_err": float(np.sqrt(1.0 / np.sum(w))),
        })

    return pd.DataFrame(rows).sort_values("time")


def prepare_curves(raw_dir: Path, curves_dir: Path) -> pd.DataFrame:
    curves_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    # Paper I: HST/COS 1367 and lines
    p = raw_dir / "J_ApJ_806_128_table2.dat"
    if p.exists():
        colspecs = [
            (0, 11), (12, 17), (18, 22), (23, 28), (29, 33),
            (34, 38), (39, 43), (44, 48), (49, 53), (54, 65),
            (66, 71), (72, 76), (77, 82), (83, 87)
        ]
        names = [
            "HJD130", "F1367", "e_F1367", "FLya", "e_FLya",
            "FNV", "e_FNV", "FSiIV", "e_FSiIV", "HJD160",
            "FCIV", "e_FCIV", "FHeII", "e_FHeII"
        ]
        df = pd.read_fwf(p, colspecs=colspecs, names=names, comment="#").apply(pd.to_numeric, errors="coerce")

        manifest_rows.append(save_curve(pd.DataFrame({
            "time": normalize_hjd_to_2400000(df["HJD130"]),
            "flux": df["F1367"],
            "flux_err": df["e_F1367"],
        }), curves_dir / "hst_cos_1367_paper1.csv"))

        for col, err, outname, tcol in [
            ("FLya", "e_FLya", "line_Lya.csv", "HJD130"),
            ("FNV", "e_FNV", "line_NV1240.csv", "HJD130"),
            ("FSiIV", "e_FSiIV", "line_SiIV1400.csv", "HJD130"),
            ("FCIV", "e_FCIV", "line_CIV1549.csv", "HJD160"),
            ("FHeII", "e_FHeII", "line_HeII1640.csv", "HJD160"),
        ]:
            manifest_rows.append(save_curve(pd.DataFrame({
                "time": normalize_hjd_to_2400000(df[tcol]),
                "flux": df[col],
                "flux_err": df[err],
            }), curves_dir / outname))

    # Paper II: Swift UVOT
    p = raw_dir / "J_ApJ_806_129_table2.dat"
    if p.exists():
        colspecs = [(0, 11), (12, 16), (17, 22), (23, 28)]
        names = ["HJD", "Filt", "Flux", "e_Flux"]
        df = pd.read_fwf(p, colspecs=colspecs, names=names, comment="#")
        df["HJD"] = normalize_hjd_to_2400000(df["HJD"])
        df["Flux"] = pd.to_numeric(df["Flux"], errors="coerce")
        df["e_Flux"] = pd.to_numeric(df["e_Flux"], errors="coerce")
        df["Filt"] = df["Filt"].astype(str).str.strip()

        for filt, g in df.groupby("Filt"):
            if filt == "" or filt.lower() == "nan":
                continue
            manifest_rows.append(save_curve(pd.DataFrame({
                "time": g["HJD"],
                "flux": g["Flux"],
                "flux_err": g["e_Flux"],
            }), curves_dir / f"swift_{filt}.csv"))

    # Paper II: Swift XRT
    p = raw_dir / "J_ApJ_806_129_table3.dat"
    if p.exists():
        colspecs = [(0, 11), (12, 17), (18, 23), (24, 29), (30, 35)]
        names = ["HJD", "HX", "e_HX", "SX", "e_SX"]
        df = pd.read_fwf(p, colspecs=colspecs, names=names, comment="#").apply(pd.to_numeric, errors="coerce")
        df["HJD"] = normalize_hjd_to_2400000(df["HJD"])

        manifest_rows.append(save_curve(pd.DataFrame({
            "time": df["HJD"],
            "flux": df["HX"],
            "flux_err": df["e_HX"],
        }), curves_dir / "swift_xrt_hard_0p8_10keV.csv"))

        manifest_rows.append(save_curve(pd.DataFrame({
            "time": df["HJD"],
            "flux": df["SX"],
            "flux_err": df["e_SX"],
        }), curves_dir / "swift_xrt_soft_0p3_0p8keV.csv"))

    # Paper III: optical
    p = raw_dir / "J_ApJ_821_56_table3.dat"
    if p.exists():
        colspecs = [(0, 1), (2, 13), (14, 21), (22, 28), (29, 35), (36, 46), (47, 55)]
        names = ["Filt", "HJD", "Flux", "e_Flux", "Tel", "dCTS", "e_dCTS"]
        df = pd.read_fwf(p, colspecs=colspecs, names=names, comment="#")
        df["HJD"] = normalize_hjd_to_2400000(df["HJD"])
        for c in ["Flux", "e_Flux", "dCTS", "e_dCTS"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Filt"] = df["Filt"].astype(str).str.strip()

        for filt, g in df.groupby("Filt"):
            if filt == "" or filt.lower() == "nan":
                continue

            raw_curve = pd.DataFrame({
                "time": g["HJD"],
                "flux": g["Flux"],
                "flux_err": g["e_Flux"],
            })
            manifest_rows.append(save_curve(raw_curve, curves_dir / f"opt_{filt}_raw.csv"))

            daily = weighted_bin_average(raw_curve, bin_days=0.25)
            manifest_rows.append(save_curve(daily, curves_dir / f"opt_{filt}_daily.csv"))

    # Paper III: HST continuum light curves
    p = raw_dir / "J_ApJ_821_56_table4.dat"
    if p.exists():
        colspecs = [(0, 6), (7, 15), (16, 21), (22, 26)]
        names = ["lambda", "HJD", "Flux", "e_Flux"]
        df = pd.read_fwf(p, colspecs=colspecs, names=names, comment="#").apply(pd.to_numeric, errors="coerce")
        df["HJD"] = normalize_hjd_to_2400000(df["HJD"])

        for lam, g in df.groupby("lambda"):
            if not np.isfinite(lam):
                continue
            lam_int = int(round(float(lam)))
            manifest_rows.append(save_curve(pd.DataFrame({
                "time": g["HJD"],
                "flux": g["Flux"],
                "flux_err": g["e_Flux"],
            }), curves_dir / f"hst_cos_{lam_int}.csv"))

    man = pd.DataFrame(manifest_rows).sort_values("file")
    man.to_csv(curves_dir.parent / "curves_manifest.csv", index=False)
    return man


# ---------- Задержки ----------

def flic_delay_days(mass_msun: float, z: float, alpha: float) -> tuple[float, float]:
    rs = 2.0 * G * mass_msun * M_SUN / C**2
    tau_source = alpha * (rs / C) * math.log(rs / L_PLANCK) / DAY
    tau_obs = (1.0 + z) * tau_source
    return tau_source, tau_obs


def write_delay_table(results_dir: Path, mass_msun: float = 6.5e7, z: float = 0.017175, alpha: float = 0.75) -> pd.DataFrame:
    rows = []
    for name, a in [
        ("side_2over3", 2.0 / 3.0),
        ("canonical_3over4", alpha),
        ("side_4over3", 4.0 / 3.0),
    ]:
        src, obs = flic_delay_days(mass_msun, z, a)
        rows.append({
            "name": name,
            "mass_msun": mass_msun,
            "z": z,
            "alpha": a,
            "tau_source_days": src,
            "tau_source_hours": src * 24.0,
            "tau_obs_days": obs,
            "tau_obs_hours": obs * 24.0,
            "two_tau_obs_days": 2.0 * obs,
            "two_tau_obs_hours": 48.0 * obs,
        })
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "ngc5548_flic_delays_v3.csv", index=False)
    return df


# ---------- DCF lag search ----------

def standardize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.nanmean(y)
    sd = np.nanstd(y)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Нельзя нормировать постоянную или повреждённую кривую.")
    return (y - mu) / sd


@dataclass
class PairPrecompute:
    lag_pairs: np.ndarray
    driver_z: np.ndarray
    response_time: np.ndarray


def precompute_pair(driver: pd.DataFrame, response: pd.DataFrame) -> PairPrecompute:
    td = driver["time"].to_numpy(float)
    tr = response["time"].to_numpy(float)
    lag_pairs = tr[None, :] - td[:, None]
    return PairPrecompute(
        lag_pairs=lag_pairs,
        driver_z=standardize(driver["flux"].to_numpy(float)),
        response_time=tr,
    )


def dcf_score_from_response(pre: PairPrecompute, response_flux: np.ndarray, lags: np.ndarray, bin_width: float) -> pd.DataFrame:
    yr = standardize(np.asarray(response_flux, dtype=float))
    prod = pre.driver_z[:, None] * yr[None, :]
    half = 0.5 * bin_width

    rows = []
    for lag in lags:
        m = (pre.lag_pairs >= lag - half) & (pre.lag_pairs < lag + half)
        n = int(m.sum())
        if n >= 3:
            vals = prod[m]
            score = float(np.nanmean(vals))
            scatter = float(np.nanstd(vals, ddof=1)) if n > 3 else np.nan
        else:
            score = np.nan
            scatter = np.nan
        rows.append({"lag_days": float(lag), "dcf_score": score, "pair_scatter": scatter, "n_pairs": n})
    return pd.DataFrame(rows)


def summarize_target(
    ccf: pd.DataFrame,
    target: float,
    name: str,
    peak_window: float,
    side_min: float,
    side_max: float,
) -> dict:
    x = ccf[np.isfinite(ccf["dcf_score"])].copy()

    if len(x) == 0:
        return {
            "target_name": name,
            "tau_target_days": target,
            "tau_at_target_grid_days": np.nan,
            "tau_peak_days": np.nan,
            "lambda_target_grid": np.nan,
            "lambda_peak": np.nan,
            "dcf_at_target": np.nan,
            "dcf_peak": np.nan,
            "local_z_target": np.nan,
            "local_z_peak": np.nan,
            "n_pairs_at_target": 0,
            "n_pairs_at_peak": 0,
            "n_sideband": 0,
        }

    i0 = int(np.nanargmin(np.abs(x["lag_days"].to_numpy() - target)))
    row0 = x.iloc[i0]

    near = x[(x["lag_days"] >= target - peak_window) & (x["lag_days"] <= target + peak_window)]
    if len(near) > 0:
        rowp = near.iloc[int(np.nanargmax(near["dcf_score"].to_numpy()))]
    else:
        rowp = row0

    dist = np.abs(x["lag_days"] - target)
    side = x[(dist >= side_min) & (dist <= side_max)]
    side_score = side["dcf_score"].to_numpy(float)

    if len(side_score) >= 10 and np.nanstd(side_score, ddof=1) > 0:
        mu = float(np.nanmean(side_score))
        sd = float(np.nanstd(side_score, ddof=1))
        z_target = (float(row0["dcf_score"]) - mu) / sd
        z_peak = (float(rowp["dcf_score"]) - mu) / sd
    else:
        z_target = np.nan
        z_peak = np.nan

    return {
        "target_name": name,
        "tau_target_days": target,
        "tau_at_target_grid_days": float(row0["lag_days"]),
        "tau_peak_days": float(rowp["lag_days"]),
        "lambda_target_grid": float(row0["lag_days"]) / target,
        "lambda_peak": float(rowp["lag_days"]) / target,
        "dcf_at_target": float(row0["dcf_score"]),
        "dcf_peak": float(rowp["dcf_score"]),
        "local_z_target": z_target,
        "local_z_peak": z_peak,
        "n_pairs_at_target": int(row0["n_pairs"]),
        "n_pairs_at_peak": int(rowp["n_pairs"]),
        "n_sideband": int(len(side_score)),
    }


def run_lag_pair(
    driver_path: Path,
    response_path: Path,
    out_prefix: Path,
    lag_min: float = -2.0,
    lag_max: float = 4.0,
    lag_step: float = 0.01,
    bin_width: float = 0.08,
    peak_window: float = 0.15,
    side_min: float = 0.25,
    side_max: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    driver = load_curve(driver_path)
    response = load_curve(response_path)
    pre = precompute_pair(driver, response)
    lags = np.arange(lag_min, lag_max + 0.5 * lag_step, lag_step)

    ccf = dcf_score_from_response(pre, response["flux"].to_numpy(float), lags, bin_width)

    rows = [
        summarize_target(ccf, TAU_PRIMARY, "primary_14p4h", peak_window, side_min, side_max),
        summarize_target(ccf, TAU_DOUBLE, "double_28p8h", peak_window, side_min, side_max),
    ]
    summary = pd.DataFrame(rows)
    summary.insert(0, "driver", driver_path.name)
    summary.insert(1, "response", response_path.name)
    summary.insert(2, "bin_width_days", bin_width)
    summary.insert(3, "peak_window_days", peak_window)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ccf.to_csv(Path(str(out_prefix) + ".ccf.csv"), index=False)
    summary.to_csv(Path(str(out_prefix) + ".summary.csv"), index=False)
    return ccf, summary


def run_baseline(
    curves_dir: Path,
    results_dir: Path,
    driver_name: str,
    response_names: list[str],
    bin_width: float = 0.08,
    peak_window: float = 0.15,
) -> pd.DataFrame:
    driver_path = curves_dir / driver_name
    if not driver_path.exists():
        raise FileNotFoundError(f"Нет драйвера: {driver_path}")

    summaries = []
    for resp_name in response_names:
        resp_path = curves_dir / resp_name
        if not resp_path.exists():
            print(f"[skip] нет отклика: {resp_path}")
            continue

        out_prefix = results_dir / f"{driver_path.stem}_TO_{resp_path.stem}_v3"
        _, s = run_lag_pair(
            driver_path,
            resp_path,
            out_prefix,
            bin_width=bin_width,
            peak_window=peak_window,
        )
        summaries.append(s)

    if summaries:
        out = pd.concat(summaries, ignore_index=True)
    else:
        out = pd.DataFrame()

    out.to_csv(results_dir / "baseline_summary_v3.csv", index=False)
    return out


# ---------- Нулевые тесты ----------

def circular_shift(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(1, len(y)))
    return np.roll(y, k)


def permute_flux(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return y[rng.permutation(len(y))]


def ou_like_surrogate(t: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mu = float(np.nanmean(y))
    sd = float(np.nanstd(y))
    total = float(np.nanmax(t) - np.nanmin(t))
    tau = max(1.0, 0.2 * total)

    sim = np.empty_like(y, dtype=float)
    sim[0] = rng.normal(mu, sd)

    for i in range(1, len(y)):
        dt = max(0.0, float(t[i] - t[i - 1]))
        a = math.exp(-dt / tau)
        sim[i] = mu + a * (sim[i - 1] - mu) + rng.normal(0.0, sd * math.sqrt(max(0.0, 1.0 - a * a)))

    return sim


def summarize_from_flux(
    pre: PairPrecompute,
    response_flux: np.ndarray,
    lags: np.ndarray,
    bin_width: float,
    peak_window: float,
    side_min: float,
    side_max: float,
) -> list[dict]:
    ccf = dcf_score_from_response(pre, response_flux, lags, bin_width)
    return [
        summarize_target(ccf, TAU_PRIMARY, "primary_14p4h", peak_window, side_min, side_max),
        summarize_target(ccf, TAU_DOUBLE, "double_28p8h", peak_window, side_min, side_max),
    ]


def empirical_p_value(null_values: np.ndarray, real_value: float) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if not np.isfinite(real_value) or len(null_values) == 0:
        return np.nan
    return float((1 + np.sum(null_values >= real_value)) / (len(null_values) + 1))


def run_null_tests(
    curves_dir: Path,
    results_dir: Path,
    pair: str,
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
) -> dict[str, pd.DataFrame]:
    if modes is None:
        modes = ["shift", "permute", "ou"]

    if ":" not in pair:
        raise ValueError("--null-pair должен быть формата driver.csv:response.csv")

    driver_name, response_name = pair.split(":", 1)
    driver_path = curves_dir / driver_name
    response_path = curves_dir / response_name

    driver = load_curve(driver_path)
    response = load_curve(response_path)

    pre = precompute_pair(driver, response)
    y = response["flux"].to_numpy(float)
    t = response["time"].to_numpy(float)
    lags = np.arange(lag_min, lag_max + 0.5 * lag_step, lag_step)

    real_rows = summarize_from_flux(pre, y, lags, bin_width, peak_window, side_min, side_max)
    real_df = pd.DataFrame(real_rows)
    real_df.insert(0, "driver", driver_name)
    real_df.insert(1, "response", response_name)
    real_df.insert(2, "bin_width_days", bin_width)
    real_df.insert(3, "peak_window_days", peak_window)

    if real_df["n_pairs_at_peak"].fillna(0).sum() == 0:
        raise RuntimeError("В реальной паре нет пар точек около целевых лагов. Проверь временные шкалы и bin_width.")

    rng = np.random.default_rng(seed)
    null_rows = []

    for mode in modes:
        print(f"[null] {pair} mode={mode}")
        for i in range(n_null):
            if mode == "shift":
                yy = circular_shift(y, rng)
            elif mode == "permute":
                yy = permute_flux(y, rng)
            elif mode == "ou":
                yy = ou_like_surrogate(t, y, rng)
            else:
                raise ValueError(f"Неизвестный режим null: {mode}")

            for r in summarize_from_flux(pre, yy, lags, bin_width, peak_window, side_min, side_max):
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
                "null_mode": mode,
                "real_local_z_target": real["local_z_target"],
                "real_local_z_peak": real["local_z_peak"],
                "real_dcf_at_target": real["dcf_at_target"],
                "real_dcf_peak": real["dcf_peak"],
                "p_local_z_target": empirical_p_value(sub["local_z_target"].to_numpy(float), real["local_z_target"]),
                "p_local_z_peak": empirical_p_value(sub["local_z_peak"].to_numpy(float), real["local_z_peak"]),
                "p_dcf_at_target": empirical_p_value(sub["dcf_at_target"].to_numpy(float), real["dcf_at_target"]),
                "p_dcf_peak": empirical_p_value(sub["dcf_peak"].to_numpy(float), real["dcf_peak"]),
                "null_local_z_peak_mean": float(np.nanmean(sub["local_z_peak"])),
                "null_local_z_peak_std": float(np.nanstd(sub["local_z_peak"], ddof=1)),
                "null_local_z_peak_max": float(np.nanmax(sub["local_z_peak"])),
            })

    pval_df = pd.DataFrame(p_rows)
    null_summary = null_df.groupby(["null_mode", "target_name"]).agg(
        count=("local_z_peak", "count"),
        local_z_target_mean=("local_z_target", "mean"),
        local_z_target_std=("local_z_target", "std"),
        local_z_target_max=("local_z_target", "max"),
        local_z_peak_mean=("local_z_peak", "mean"),
        local_z_peak_std=("local_z_peak", "std"),
        local_z_peak_max=("local_z_peak", "max"),
    ).reset_index()

    safe_pair = f"{Path(driver_name).stem}_TO_{Path(response_name).stem}"
    prefix = results_dir / f"null_{safe_pair}_v3"
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
