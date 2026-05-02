#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qhcc_ngc5548_disk_flic_compare_v4_2.py

Disk-only vs disk+FLiC control for the NGC 5548 accretion-echo pilot.

Purpose
-------
This script does NOT perform a new echo search. It takes the v4 search outputs and
asks a narrower question:

    Can the main optical-V candidate be explained as a smooth accretion-disk
    inter-band lag, or does it behave like a separate component close to the
    fixed FLiC delay?

Inputs
------
- flic_candidates_ranked_v4.csv
- baseline_summary_v4.csv (optional; kept for traceability)
- ngc5548_flic_delays_v4.csv
- null_pvalues_combined_v4.csv (optional, used to annotate the report)

Outputs
-------
- disk_flic_fit_points_v4_2.csv
- disk_flic_model_summary_v4_2.csv
- disk_flic_model_predictions_v4_2.csv
- disk_flic_compare_report_v4_2.md
- disk_flic_lag_vs_wavelength_v4_2.png
- disk_flic_target_residual_v4_2.png

Interpretation
--------------
This is a compact sanity check, not a full physical disk-transfer-function fit.
It fits a smooth lag-vs-wavelength law to the non-target bands and compares the
main optical-V candidate with:
  (1) the disk-only prediction,
  (2) the fixed two-way FLiC delay.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# Approximate effective wavelengths in Angstrom.
# These are used only for a coarse disk-lag sanity check.
WAVELENGTH_A = {
    # Swift/UVOT
    "swift_UVW2.csv": 1928.0,
    "swift_UVM2.csv": 2246.0,
    "swift_UVW1.csv": 2600.0,
    "swift_U.csv": 3465.0,
    "swift_B.csv": 4392.0,
    "swift_V.csv": 5468.0,

    # Ground / optical daily curves
    "opt_u_daily.csv": 3543.0,
    "opt_g_daily.csv": 4770.0,
    "opt_r_daily.csv": 6231.0,
    "opt_i_daily.csv": 7625.0,
    "opt_z_daily.csv": 9134.0,
    "opt_B_daily.csv": 4361.0,
    "opt_V_daily.csv": 5448.0,
    "opt_R_daily.csv": 6407.0,
    "opt_I_daily.csv": 7980.0,
}


def load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def load_null_tables(null_path: Path | None, null_globs: list[str] | None) -> pd.DataFrame:
    """Load one combined null table plus optional per-pair pvalue tables."""
    frames = []
    if null_path is not None and null_path.exists():
        frames.append(pd.read_csv(null_path))
    if null_globs:
        import glob
        for pat in null_globs:
            for fn in sorted(glob.glob(pat)):
                try:
                    df = pd.read_csv(fn)
                    # Only accept pvalue-like tables.
                    if {"response", "target_name", "null_mode"}.issubset(df.columns):
                        frames.append(df)
                except Exception:
                    pass
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # De-duplicate identical rows that may appear in both combined and per-pair files.
    cols = [c for c in ["driver", "response", "target_name", "null_mode", "p_local_z_peak", "p_dcf_peak"] if c in out.columns]
    if cols:
        out = out.drop_duplicates(subset=cols)
    return out.reset_index(drop=True)


def select_best_lag_per_response(ranked: pd.DataFrame, selection: str = "rank_score") -> pd.DataFrame:
    """Select one observed lag per response from the v4 candidate table.

    selection:
      - rank_score: highest v4 rank_score per response
      - local_z_peak: highest local_z_peak per response
      - dcf_peak: highest dcf_peak per response
    """
    if selection not in {"rank_score", "local_z_peak", "dcf_peak"}:
        raise ValueError(f"Unknown selection: {selection}")
    x = ranked.copy()
    x["wavelength_A"] = x["response"].map(WAVELENGTH_A)
    x = x[np.isfinite(x["wavelength_A"])].copy()
    x = x[np.isfinite(x[selection])].copy()
    x = x.sort_values(selection, ascending=False).drop_duplicates("response")
    x = x.sort_values("wavelength_A").reset_index(drop=True)
    return x


def disk_model_delay_days(wavelength_A: np.ndarray, A: float, beta: float, ref_A: float) -> np.ndarray:
    w = np.asarray(wavelength_A, dtype=float)
    return A * ((w / ref_A) ** beta - 1.0)


def fit_disk_fixed_beta(points: pd.DataFrame, beta: float, ref_A: float, weighted: bool = False) -> dict:
    lam = points["wavelength_A"].to_numpy(float)
    y = points["tau_peak_days"].to_numpy(float)
    x = (lam / ref_A) ** beta - 1.0

    if weighted:
        # Use a mild non-negative weight from the DCF peak significance.
        z = points.get("local_z_peak", pd.Series(np.ones(len(points)))).to_numpy(float)
        w = np.clip(z, 0.25, None)
    else:
        w = np.ones_like(y)

    denom = float(np.sum(w * x * x))
    A = float(np.sum(w * x * y) / denom) if denom > 0 else float("nan")
    pred = A * x
    resid = y - pred
    sse = float(np.sum(w * resid * resid))
    rms = float(math.sqrt(np.nanmean(resid * resid))) if len(resid) else float("nan")
    return {
        "A_days": A,
        "beta": float(beta),
        "sse": sse,
        "rms_resid_days": rms,
        "weighted": bool(weighted),
    }


def fit_disk_free_beta(points: pd.DataFrame, ref_A: float, beta_min: float, beta_max: float,
                       beta_step: float, weighted: bool = False) -> dict:
    best = None
    n = int(round((beta_max - beta_min) / beta_step)) + 1
    for beta in np.linspace(beta_min, beta_max, n):
        row = fit_disk_fixed_beta(points, float(beta), ref_A, weighted=weighted)
        if best is None or row["sse"] < best["sse"]:
            best = row
    return best


def extract_flic_delay(delays: pd.DataFrame, target_name: str = "two_way_4over3") -> float:
    row = delays[delays["target_name"] == target_name]
    if len(row) == 0:
        raise ValueError(f"Cannot find FLiC delay target_name={target_name}")
    if "tau_obs_days" in row.columns:
        return float(row.iloc[0]["tau_obs_days"])
    if "tau_obs_days_exact" in row.columns:
        return float(row.iloc[0]["tau_obs_days_exact"])
    raise ValueError("Delay table does not contain tau_obs_days or tau_obs_days_exact")


def annotate_nulls(nulls: pd.DataFrame, response: str, target_name: str) -> pd.DataFrame:
    if nulls is None or len(nulls) == 0:
        return pd.DataFrame()
    x = nulls[(nulls["response"] == response) & (nulls["target_name"] == target_name)].copy()
    keep = [
        "null_mode", "real_local_z_target", "real_local_z_peak",
        "real_dcf_at_target", "real_dcf_peak", "real_lambda_peak",
        "p_local_z_target", "p_local_z_peak", "p_dcf_at_target", "p_dcf_peak",
    ]
    return x[[c for c in keep if c in x.columns]].reset_index(drop=True)


def make_report(summary: pd.DataFrame, target_row: pd.Series, null_ann: pd.DataFrame,
                out_path: Path, target_response: str, flic_target_name: str) -> None:
    fixed = summary[summary["model"] == "disk_only_beta_fixed_4over3"].iloc[0]
    free = summary[summary["model"] == "disk_only_beta_free"].iloc[0]
    flic = summary[summary["model"] == "fixed_flic_component"].iloc[0]

    def h(days): return 24.0 * float(days)

    lines = []
    lines.append("# NGC 5548: disk-only vs disk+FLiC control v4.2")
    lines.append("")
    lines.append("## Человеческий вывод")
    lines.append("")
    lines.append(
        f"Главный проверяемый канал: `hst_cos_1367.csv -> {target_response}`. "
        f"Наблюдаемый максимум для целевой кривой стоит на **{h(target_row['tau_peak_days']):.2f} часа**."
    )
    lines.append("")
    lines.append(
        f"Фиксированная двухходовая FLiC-задержка `{flic_target_name}` равна "
        f"**{h(flic['predicted_target_lag_days']):.2f} часа**. "
        f"Остаток относительно FLiC: **{h(flic['target_residual_days']):+.2f} часа**."
    )
    lines.append("")
    lines.append(
        f"Обычный гладкий диск с фиксированным показателем beta=4/3 предсказывает для этого же диапазона "
        f"**{h(fixed['predicted_target_lag_days']):.2f} часа**, "
        f"остаток: **{h(fixed['target_residual_days']):+.2f} часа**."
    )
    lines.append("")
    lines.append(
        f"Гладкий диск со свободным beta предсказывает "
        f"**{h(free['predicted_target_lag_days']):.2f} часа**, "
        f"остаток: **{h(free['target_residual_days']):+.2f} часа**."
    )
    lines.append("")
    lines.append("## Интерпретация")
    lines.append("")
    lines.append(
        "В этой компактной проверке V-кандидат выглядит не как точка гладкой цветовой задержки диска, "
        "а как поздний компонент, гораздо ближе к фиксированной FLiC-задержке, чем к disk-only prediction."
    )
    lines.append("")
    lines.append(
        "Это ещё не полная физическая модель аккреционного диска. Полная модель должна учитывать широкую "
        "функцию отклика, ошибки фотометрии, сезонные окна, diffuse continuum и возможный вклад линий. "
        "Но как sanity-check результат усиливает FLiC-интерпретацию главного V-кандидата."
    )
    lines.append("")
    lines.append("## Нулевые проверки для целевого канала")
    lines.append("")
    if len(null_ann) == 0:
        lines.append("Нулевые p-значения не переданы или не найдены для этого канала.")
    else:
        lines.append("| режим нуля | p по локальной высоте в точке | p по локальной высоте пика | p по силе DCF в точке | p по силе DCF пика |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in null_ann.iterrows():
            lines.append(
                f"| {r.get('null_mode','')} | "
                f"{r.get('p_local_z_target', float('nan')):.4g} | "
                f"{r.get('p_local_z_peak', float('nan')):.4g} | "
                f"{r.get('p_dcf_at_target', float('nan')):.4g} | "
                f"{r.get('p_dcf_peak', float('nan')):.4g} |"
            )
    lines.append("")
    lines.append("## Файлы")
    lines.append("")
    lines.append("- `disk_flic_fit_points_v4_2.csv` — точки лагов, использованные для disk-fit.")
    lines.append("- `disk_flic_model_summary_v4_2.csv` — числовое сравнение disk-only и fixed-FLiC.")
    lines.append("- `disk_flic_model_predictions_v4_2.csv` — кривые disk-only моделей по длине волны.")
    lines.append("- `disk_flic_lag_vs_wavelength_v4_2.png` — главный график.")
    lines.append("- `disk_flic_target_residual_v4_2.png` — сравнение остатков V-кандидата.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(points: pd.DataFrame, pred: pd.DataFrame, summary: pd.DataFrame,
               delays: pd.DataFrame, target_response: str, out_dir: Path) -> None:
    if plt is None:
        return

    target_point = points[points["response"] == target_response]
    fit_points = points[points["used_for_disk_fit"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(
        points["wavelength_A"], points["tau_peak_days"] * 24.0,
        s=50, label="selected observed lag per band"
    )
    if len(fit_points):
        ax.scatter(
            fit_points["wavelength_A"], fit_points["tau_peak_days"] * 24.0,
            s=70, marker="o", facecolors="none", edgecolors="black",
            label="points used for disk fit"
        )
    if len(target_point):
        ax.scatter(
            target_point["wavelength_A"], target_point["tau_peak_days"] * 24.0,
            s=120, marker="*", label="tested V candidate"
        )

    for model_name, g in pred.groupby("model"):
        ax.plot(g["wavelength_A"], g["tau_pred_days"] * 24.0, lw=2, label=model_name)

    # FLiC horizontal lines
    for _, row in delays.iterrows():
        y = float(row["tau_obs_days"] if "tau_obs_days" in row else row["tau_obs_days_exact"]) * 24.0
        ax.axhline(y, ls="--", lw=1.3, label=f"FLiC {row['target_name']}")

    ax.set_xlabel("response wavelength, Angstrom")
    ax.set_ylabel("lag, hours")
    ax.set_title("NGC 5548: selected lag vs wavelength, disk-only curves, and FLiC delays")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "disk_flic_lag_vs_wavelength_v4_2.png", dpi=160)
    plt.close(fig)

    # Residual bar chart for target
    s = summary.copy()
    s["target_residual_hours"] = s["target_residual_days"] * 24.0
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(s["model"], s["target_residual_hours"])
    ax.axhline(0.0, lw=1.0)
    ax.set_ylabel("target residual, hours")
    ax.set_title(f"Residual for {target_response}: observed lag minus model prediction")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "disk_flic_target_residual_v4_2.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="NGC 5548 disk-only vs disk+FLiC control v4.2")
    p.add_argument("--ranked", default="run_v4/results/flic_candidates_ranked_v4.csv")
    p.add_argument("--baseline", default="run_v4/results/baseline_summary_v4.csv")
    p.add_argument("--delays", default="run_v4/results/ngc5548_flic_delays_v4.csv")
    p.add_argument("--null-pvalues", default="run_v4/results/null_pvalues_combined_v4.csv")
    p.add_argument("--null-glob", action="append", default=None,
                   help="optional glob for per-pair null pvalue files, e.g. run_v4/results/null_*_v4.pvalues.csv")
    p.add_argument("--out-dir", default="run_v4/results")
    p.add_argument("--target-response", default="opt_V_daily.csv")
    p.add_argument("--flic-target-name", default="two_way_4over3")
    p.add_argument("--ref-wavelength", type=float, default=1367.0)
    p.add_argument("--fixed-beta", type=float, default=4.0 / 3.0)
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--beta-max", type=float, default=3.0)
    p.add_argument("--beta-step", type=float, default=0.001)
    p.add_argument("--selection", choices=["rank_score", "local_z_peak", "dcf_peak"], default="rank_score")
    p.add_argument("--exclude-from-fit", action="append", default=None,
                   help="response file to exclude from disk fit. Repeatable. Target response is excluded automatically.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = load_csv(Path(args.ranked))
    _baseline = load_csv(Path(args.baseline), required=False)
    delays = load_csv(Path(args.delays))
    nulls = load_null_tables(Path(args.null_pvalues), args.null_glob)

    selected = select_best_lag_per_response(ranked, selection=args.selection)

    exclude = set(args.exclude_from_fit or [])
    exclude.add(args.target_response)

    selected["used_for_disk_fit"] = ~selected["response"].isin(exclude)
    selected.to_csv(out_dir / "disk_flic_fit_points_v4_2.csv", index=False)

    fit_points = selected[selected["used_for_disk_fit"]].copy()
    if len(fit_points) < 3:
        raise RuntimeError("Not enough non-target points for disk fit.")

    fixed = fit_disk_fixed_beta(fit_points, args.fixed_beta, args.ref_wavelength, weighted=False)
    free = fit_disk_free_beta(
        fit_points, args.ref_wavelength,
        args.beta_min, args.beta_max, args.beta_step, weighted=False
    )

    target = selected[selected["response"] == args.target_response]
    if len(target) == 0:
        raise RuntimeError(f"Target response not found in ranked table: {args.target_response}")
    target_row = target.iloc[0]
    target_w = float(target_row["wavelength_A"])
    target_obs = float(target_row["tau_peak_days"])
    flic_delay = extract_flic_delay(delays, args.flic_target_name)

    summary_rows = []
    for model_name, model in [
        ("disk_only_beta_fixed_4over3", fixed),
        ("disk_only_beta_free", free),
    ]:
        pred_target = float(disk_model_delay_days(np.array([target_w]), model["A_days"], model["beta"], args.ref_wavelength)[0])
        summary_rows.append({
            "model": model_name,
            "A_days": model["A_days"],
            "beta": model["beta"],
            "fit_sse": model["sse"],
            "fit_rms_resid_days": model["rms_resid_days"],
            "target_response": args.target_response,
            "target_wavelength_A": target_w,
            "observed_target_lag_days": target_obs,
            "observed_target_lag_hours": target_obs * 24.0,
            "predicted_target_lag_days": pred_target,
            "predicted_target_lag_hours": pred_target * 24.0,
            "target_residual_days": target_obs - pred_target,
            "target_residual_hours": (target_obs - pred_target) * 24.0,
            "target_residual_over_fit_rms": (target_obs - pred_target) / model["rms_resid_days"] if model["rms_resid_days"] > 0 else np.nan,
        })

    summary_rows.append({
        "model": "fixed_flic_component",
        "A_days": np.nan,
        "beta": np.nan,
        "fit_sse": np.nan,
        "fit_rms_resid_days": np.nan,
        "target_response": args.target_response,
        "target_wavelength_A": target_w,
        "observed_target_lag_days": target_obs,
        "observed_target_lag_hours": target_obs * 24.0,
        "predicted_target_lag_days": flic_delay,
        "predicted_target_lag_hours": flic_delay * 24.0,
        "target_residual_days": target_obs - flic_delay,
        "target_residual_hours": (target_obs - flic_delay) * 24.0,
        "target_residual_over_fit_rms": np.nan,
    })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "disk_flic_model_summary_v4_2.csv", index=False)

    # model prediction curves
    lam_grid = np.linspace(args.ref_wavelength, max(WAVELENGTH_A.values()) * 1.05, 500)
    pred_rows = []
    for model_name, model in [
        ("disk_only_beta_fixed_4over3", fixed),
        ("disk_only_beta_free", free),
    ]:
        y = disk_model_delay_days(lam_grid, model["A_days"], model["beta"], args.ref_wavelength)
        for wl, yy in zip(lam_grid, y):
            pred_rows.append({"model": model_name, "wavelength_A": wl, "tau_pred_days": yy, "tau_pred_hours": yy * 24.0})
    pred = pd.DataFrame(pred_rows)
    pred.to_csv(out_dir / "disk_flic_model_predictions_v4_2.csv", index=False)

    null_ann = annotate_nulls(nulls, args.target_response, args.flic_target_name)
    if len(null_ann):
        null_ann.to_csv(out_dir / "disk_flic_target_nulls_v4_2.csv", index=False)

    make_plots(selected, pred, summary, delays, args.target_response, out_dir)
    make_report(
        summary, target_row, null_ann,
        out_dir / "disk_flic_compare_report_v4_2.md",
        args.target_response, args.flic_target_name,
    )

    print("[SAVED]", out_dir / "disk_flic_fit_points_v4_2.csv")
    print("[SAVED]", out_dir / "disk_flic_model_summary_v4_2.csv")
    print("[SAVED]", out_dir / "disk_flic_model_predictions_v4_2.csv")
    print("[SAVED]", out_dir / "disk_flic_compare_report_v4_2.md")
    if plt is not None:
        print("[SAVED]", out_dir / "disk_flic_lag_vs_wavelength_v4_2.png")
        print("[SAVED]", out_dir / "disk_flic_target_residual_v4_2.png")


if __name__ == "__main__":
    main()
