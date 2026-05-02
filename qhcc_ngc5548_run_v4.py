#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qhcc_ngc5548_run_v4.py

Unified runner for the v4 NGC 5548 QHCC/FLiC accretion-echo pilot.

Example baseline:
python qhcc_ngc5548_run_v4.py --work-dir run_v4 --download --prepare --diagnose --baseline --rank

Example null tests:
python qhcc_ngc5548_run_v4.py --work-dir run_v4 --null \
  --null-pair hst_cos_1367.csv:opt_I_daily.csv --n-null 1000 \
  --null-mode shift --null-mode ou --null-mode false_lambda
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd

from qhcc_ngc5548_core_v4 import (
    ensure_dirs,
    download_all,
    prepare_curves,
    write_manifest,
    write_delay_table,
    build_targets_from_config,
    run_baseline,
    run_lag_pair,
    run_null_tests,
    aggregate_baseline_candidates,
)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def print_table(df: pd.DataFrame, max_rows: int = 50) -> None:
    if df is None or len(df) == 0:
        print("[empty]")
        return
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 120, "display.width", 260):
        print(df.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default="run_v4")
    p.add_argument("--config", default="config_ngc5548_v4.json")

    p.add_argument("--download", action="store_true")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--delays", action="store_true")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--rank", action="store_true")

    p.add_argument("--driver", default=None)
    p.add_argument("--response", action="append", default=None)
    p.add_argument("--lag-pair", action="append", default=None, help="driver.csv:response.csv")

    p.add_argument("--null", action="store_true")
    p.add_argument("--null-pair", action="append", default=None, help="driver.csv:response.csv")
    p.add_argument("--n-null", type=int, default=1000)
    p.add_argument("--null-mode", action="append", choices=["shift", "permute", "ou", "false_lambda"], default=None)

    p.add_argument("--bin-width", type=float, default=0.08)
    p.add_argument("--peak-window", type=float, default=0.15)
    p.add_argument("--lag-min", type=float, default=-2.0)
    p.add_argument("--lag-max", type=float, default=4.0)
    p.add_argument("--lag-step", type=float, default=0.01)
    p.add_argument("--side-min", type=float, default=0.25)
    p.add_argument("--side-max", type=float, default=3.0)

    args = p.parse_args()

    work_dir = Path(args.work_dir)
    paths = ensure_dirs(work_dir)
    cfg = load_config(Path(args.config))
    targets = build_targets_from_config(cfg)
    strict_lambda_min = float(cfg.get("strict_lambda_min", 0.7))
    strict_lambda_max = float(cfg.get("strict_lambda_max", 1.3))

    if not any([args.download, args.prepare, args.diagnose, args.delays, args.baseline, args.rank, args.lag_pair, args.null]):
        print("[info] No action. For baseline run use: --download --prepare --diagnose --baseline --rank")
        return

    if args.download:
        print("[stage] download")
        download_all(paths["raw"], force=args.force_download)

    if args.prepare:
        print("[stage] prepare")
        man = prepare_curves(paths["raw"], paths["curves"])
        print_table(man[["file", "n", "t_min", "t_max", "t_median"]], max_rows=250)

    if args.diagnose:
        print("[stage] diagnose")
        man = write_manifest(paths["curves"], work_dir / "curves_manifest.csv")
        print_table(man[["file", "n", "t_min", "t_max", "t_median"]], max_rows=250)

    if args.delays or args.baseline or args.lag_pair or args.null:
        print("[stage] delays")
        delays = write_delay_table(paths["results"], targets=targets)
        print_table(delays, max_rows=50)

    if args.baseline:
        print("[stage] baseline")
        driver = args.driver or cfg["default_driver"]
        responses = args.response or cfg["default_responses"]
        out = run_baseline(
            paths["curves"],
            paths["results"],
            driver,
            responses,
            targets=targets,
            bin_width=args.bin_width,
            peak_window=args.peak_window,
            lag_min=args.lag_min,
            lag_max=args.lag_max,
            lag_step=args.lag_step,
            side_min=args.side_min,
            side_max=args.side_max,
        )
        print_table(out, max_rows=250)
        print("[ok]", paths["results"] / "baseline_summary_v4.csv")

    if args.rank:
        print("[stage] rank")
        ranked = aggregate_baseline_candidates(paths["results"], strict_lambda_min, strict_lambda_max)
        cols = [c for c in [
            "driver", "response", "target_name", "branch_label", "alpha", "tau_obs_days_exact",
            "tau_peak_days", "lambda_peak", "dcf_at_target", "dcf_peak", "local_z_target",
            "local_z_peak", "n_pairs_at_target", "n_pairs_at_peak", "rank_score",
        ] if c in ranked.columns]
        print_table(ranked[cols].head(40), max_rows=40)
        print("[ok]", paths["results"] / "flic_candidates_ranked_v4.csv")

    if args.lag_pair:
        print("[stage] lag-pair")
        for pair in args.lag_pair:
            if ":" not in pair:
                raise ValueError("--lag-pair must be driver.csv:response.csv")
            d, r = pair.split(":", 1)
            prefix = paths["results"] / f"{Path(d).stem}_TO_{Path(r).stem}_manual_v4"
            _, s = run_lag_pair(
                paths["curves"] / d,
                paths["curves"] / r,
                prefix,
                targets=targets,
                lag_min=args.lag_min,
                lag_max=args.lag_max,
                lag_step=args.lag_step,
                bin_width=args.bin_width,
                peak_window=args.peak_window,
                side_min=args.side_min,
                side_max=args.side_max,
            )
            print_table(s)
            print("[ok]", str(prefix) + ".summary.csv")

    if args.null:
        print("[stage] null")
        if not args.null_pair:
            raise ValueError("For --null, provide at least one --null-pair driver.csv:response.csv")
        modes = args.null_mode if args.null_mode else ["shift", "permute", "ou", "false_lambda"]
        combined = []
        for pair in args.null_pair:
            out = run_null_tests(
                paths["curves"],
                paths["results"],
                pair,
                targets=targets,
                n_null=args.n_null,
                modes=modes,
                lag_min=args.lag_min,
                lag_max=args.lag_max,
                lag_step=args.lag_step,
                bin_width=args.bin_width,
                peak_window=args.peak_window,
                side_min=args.side_min,
                side_max=args.side_max,
                strict_lambda_min=strict_lambda_min,
                strict_lambda_max=strict_lambda_max,
            )
            print(f"[p-values] {pair}")
            print_table(out["pvalues"], max_rows=80)
            combined.append(out["pvalues"])
        if combined:
            all_p = pd.concat(combined, ignore_index=True)
            all_p.to_csv(paths["results"] / "null_pvalues_combined_v4.csv", index=False)
            print("[ok]", paths["results"] / "null_pvalues_combined_v4.csv")


if __name__ == "__main__":
    main()
