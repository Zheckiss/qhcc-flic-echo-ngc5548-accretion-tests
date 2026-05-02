#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QHCC/FLiC NGC 5548 disk-lag control v4.1

Purpose:
  Test whether the strongest FLiC-like lag can be explained as an ordinary
  smooth accretion-disk reverberation trend.

Inputs:
  - flic_candidates_ranked_v4.csv from qhcc_ngc5548_run_v4.py --baseline --rank
  - optional null_pvalues_combined_v4.csv for context

Outputs:
  - disk_control_best_lags_v4_1.csv
  - disk_control_summary_v4_1.csv
  - disk_control_report_v4_1.md

No external Python packages are required.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

# Approximate effective wavelengths in Angstrom.
# Swift UVOT values are standard central wavelengths; optical values are
# approximate Johnson/Cousins/SDSS effective wavelengths for trend tests.
WAVELENGTH_A = {
    "swift_UVW2.csv": 1928.0,
    "swift_UVM2.csv": 2246.0,
    "swift_UVW1.csv": 2600.0,
    "swift_U.csv": 3465.0,
    "swift_B.csv": 4392.0,
    "swift_V.csv": 5468.0,
    "opt_u_daily.csv": 3543.0,
    "opt_B_daily.csv": 4361.0,
    "opt_g_daily.csv": 4770.0,
    "opt_V_daily.csv": 5448.0,
    "opt_r_daily.csv": 6231.0,
    "opt_R_daily.csv": 6580.0,
    "opt_i_daily.csv": 7625.0,
    "opt_I_daily.csv": 8060.0,
    "opt_z_daily.csv": 9134.0,
}

LABEL_RU = {
    "swift_UVW2.csv": "Swift/UVOT UVW2, короткий ультрафиолет",
    "swift_UVM2.csv": "Swift/UVOT UVM2, средний ультрафиолет",
    "swift_UVW1.csv": "Swift/UVOT UVW1, длинноволновый ультрафиолет",
    "swift_U.csv": "Swift/UVOT U, ближний ультрафиолет",
    "swift_B.csv": "Swift/UVOT B, синий оптический",
    "swift_V.csv": "Swift/UVOT V, видимый оптический",
    "opt_u_daily.csv": "наземный u, ближний ультрафиолет / синий край",
    "opt_B_daily.csv": "наземный B, синий оптический",
    "opt_g_daily.csv": "наземный g, зелёно-синий оптический",
    "opt_V_daily.csv": "наземный V, главный кандидат",
    "opt_r_daily.csv": "наземный r, красный оптический",
    "opt_R_daily.csv": "наземный R, красный оптический",
    "opt_i_daily.csv": "наземный i, ближний ИК край",
    "opt_I_daily.csv": "наземный I, ближний ИК край",
    "opt_z_daily.csv": "наземный z, ближний ИК край",
}


def ffloat(x: str, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def select_best_per_response(rows: List[dict]) -> List[dict]:
    best: Dict[str, dict] = {}
    for r in rows:
        resp = r.get("response", "")
        if resp not in WAVELENGTH_A:
            continue
        score = ffloat(r.get("rank_score", "nan"))
        zpeak = ffloat(r.get("local_z_peak", "nan"))
        keyscore = score if math.isfinite(score) else zpeak
        old = best.get(resp)
        if old is None:
            r2 = dict(r)
            r2["_keyscore"] = keyscore
            best[resp] = r2
        else:
            if keyscore > old.get("_keyscore", -1e99):
                r2 = dict(r)
                r2["_keyscore"] = keyscore
                best[resp] = r2
    out = []
    for resp, r in best.items():
        lam = WAVELENGTH_A[resp]
        tau_h = ffloat(r.get("tau_peak_days", "nan")) * 24.0
        out.append({
            "response": resp,
            "label_ru": LABEL_RU.get(resp, resp),
            "wavelength_A": lam,
            "best_target_name": r.get("target_name", ""),
            "branch_label": r.get("branch_label", ""),
            "tau_peak_hours": tau_h,
            "lambda_peak": ffloat(r.get("lambda_peak", "nan")),
            "dcf_peak": ffloat(r.get("dcf_peak", "nan")),
            "local_z_peak": ffloat(r.get("local_z_peak", "nan")),
            "rank_score": ffloat(r.get("rank_score", "nan")),
        })
    out.sort(key=lambda x: x["wavelength_A"])
    return out


def fit_disk_model(points: List[dict], lambda0: float, beta: float | None = None) -> dict:
    # Model: tau_hours = A * ((lambda/lambda0)^beta - 1), A >= 0.
    # If beta is None, grid-search beta and solve A analytically.
    def solve_for_beta(b: float) -> Tuple[float, float]:
        xs, ys = [], []
        for p in points:
            lam = float(p["wavelength_A"])
            y = float(p["tau_peak_hours"])
            if not (math.isfinite(lam) and math.isfinite(y) and y > 0 and lam > lambda0):
                continue
            x = (lam / lambda0) ** b - 1.0
            if x <= 0 or not math.isfinite(x):
                continue
            xs.append(x)
            ys.append(y)
        if len(xs) < 2:
            return float("nan"), float("inf")
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        A = max(0.0, sxy / sxx) if sxx > 0 else float("nan")
        err = 0.0
        for x, y in zip(xs, ys):
            err += (y - A * x) ** 2
        rmse = math.sqrt(err / len(xs))
        return A, rmse

    if beta is not None:
        A, rmse = solve_for_beta(beta)
        return {"beta": beta, "A_hours": A, "rmse_hours": rmse, "n_fit": len(points)}

    best = {"beta": float("nan"), "A_hours": float("nan"), "rmse_hours": float("inf"), "n_fit": len(points)}
    b = 0.30
    while b <= 3.0001:
        A, rmse = solve_for_beta(b)
        if rmse < best["rmse_hours"]:
            best = {"beta": b, "A_hours": A, "rmse_hours": rmse, "n_fit": len(points)}
        b += 0.01
    return best


def predict_hours(lam: float, lambda0: float, A: float, beta: float) -> float:
    return A * ((lam / lambda0) ** beta - 1.0)


def monotonic_violations(points: List[dict]) -> List[str]:
    msgs = []
    pts = sorted(points, key=lambda p: p["wavelength_A"])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            a, b = pts[i], pts[j]
            if b["tau_peak_hours"] + 1.0 < a["tau_peak_hours"]:
                msgs.append(
                    f"{b['response']} ({b['wavelength_A']:.0f} A) имеет лаг {b['tau_peak_hours']:.2f} ч, "
                    f"меньше чем {a['response']} ({a['wavelength_A']:.0f} A) с лагом {a['tau_peak_hours']:.2f} ч"
                )
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", default="flic_candidates_ranked_v4.csv")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--driver-wavelength", type=float, default=1367.0)
    ap.add_argument("--test-response", default="opt_V_daily.csv")
    args = ap.parse_args()

    ranked = Path(args.ranked)
    out_dir = Path(args.out_dir)
    rows = read_csv(ranked)
    best = select_best_per_response(rows)

    test = next((p for p in best if p["response"] == args.test_response), None)
    if test is None:
        raise SystemExit(f"No test response found: {args.test_response}")

    # Fit disk trend excluding the tested response, to ask whether the rest of the bands predict it.
    fit_points = [p for p in best if p["response"] != args.test_response]
    thin = fit_disk_model(fit_points, args.driver_wavelength, beta=4.0 / 3.0)
    free = fit_disk_model(fit_points, args.driver_wavelength, beta=None)

    lam_test = float(test["wavelength_A"])
    tau_test = float(test["tau_peak_hours"])

    pred_thin = predict_hours(lam_test, args.driver_wavelength, thin["A_hours"], thin["beta"])
    pred_free = predict_hours(lam_test, args.driver_wavelength, free["A_hours"], free["beta"])

    residual_thin = tau_test - pred_thin
    residual_free = tau_test - pred_free

    # Monotonicity on best-lag sequence.
    violations = monotonic_violations(best)
    redder_lower = []
    for p in best:
        if p["wavelength_A"] > lam_test and p["tau_peak_hours"] + 1.0 < tau_test:
            redder_lower.append(p)

    best_fields = [
        "response", "label_ru", "wavelength_A", "best_target_name", "branch_label",
        "tau_peak_hours", "lambda_peak", "dcf_peak", "local_z_peak", "rank_score",
    ]
    write_csv(out_dir / "disk_control_best_lags_v4_1.csv", best, best_fields)

    summary = [{
        "test_response": args.test_response,
        "test_label_ru": test["label_ru"],
        "test_wavelength_A": lam_test,
        "test_tau_peak_hours": tau_test,
        "test_branch": test["branch_label"],
        "thin_disk_beta": thin["beta"],
        "thin_disk_A_hours": thin["A_hours"],
        "thin_disk_rmse_hours_excluding_test": thin["rmse_hours"],
        "thin_disk_predicted_test_hours": pred_thin,
        "thin_disk_residual_test_hours": residual_thin,
        "free_beta": free["beta"],
        "free_beta_A_hours": free["A_hours"],
        "free_beta_rmse_hours_excluding_test": free["rmse_hours"],
        "free_beta_predicted_test_hours": pred_free,
        "free_beta_residual_test_hours": residual_free,
        "n_redder_bands_with_lower_lag_than_test": len(redder_lower),
        "n_monotonicity_violations": len(violations),
    }]
    summary_fields = list(summary[0].keys())
    write_csv(out_dir / "disk_control_summary_v4_1.csv", summary, summary_fields)

    verdict = []
    verdict.append("# Проверка обычного диска для NGC 5548 v4.1\n")
    verdict.append("## Что проверяется\n")
    verdict.append(
        "Обычный аккреционный диск должен давать плавную зависимость задержки от длины волны: "
        "чем краснее диапазон, тем дальше область диска и тем позже отклик. "
        "Проверка спрашивает: можно ли главный V-кандидат объяснить такой гладкой дисковой задержкой, "
        "без FLiC-компонента?\n"
    )
    verdict.append("## Главный проверяемый канал\n")
    verdict.append(
        f"- Канал: {test['label_ru']} ({args.test_response})\n"
        f"- Длина волны: {lam_test:.0f} A\n"
        f"- Найденная задержка: {tau_test:.2f} ч\n"
        f"- Ветвь в рейтинге: {test['branch_label']}\n"
    )
    verdict.append("## Модель диска\n")
    verdict.append(
        "Использована простая контрольная модель: tau(lambda) = A * ((lambda / 1367 A)^beta - 1). "
        "Это не финальная физическая модель диска, а жесткий sanity-check: если V-кандидат является обычной "
        "дисковой задержкой, остальные диапазоны должны предсказывать похожий лаг в V.\n"
    )
    verdict.append("## Численный результат\n")
    verdict.append(
        f"- Тонкий диск beta=4/3, без V в подгонке: предсказание для V = {pred_thin:.2f} ч; "
        f"остаток V = {residual_thin:.2f} ч.\n"
        f"- Свободная beta={free['beta']:.2f}, без V в подгонке: предсказание для V = {pred_free:.2f} ч; "
        f"остаток V = {residual_free:.2f} ч.\n"
        f"- Число более красных диапазонов с меньшим лагом, чем у V: {len(redder_lower)}.\n"
    )
    if redder_lower:
        verdict.append("## Главная проблема для простой дисковой интерпретации\n")
        verdict.append(
            "В обычной гладкой дисковой картине более красные диапазоны не должны устойчиво приходить раньше, "
            "чем V. Но в текущем рейтинге есть более красные диапазоны с меньшей задержкой:\n"
        )
        for p in redder_lower:
            verdict.append(f"- {p['response']}: {p['wavelength_A']:.0f} A, лаг {p['tau_peak_hours']:.2f} ч\n")
    verdict.append("## Предварительный вывод\n")
    if abs(residual_free) > 4.0 or redder_lower:
        verdict.append(
            "Главный V-кандидат плохо выглядит как часть одной гладкой дисковой задержки. "
            "Это поддерживает интерпретацию V как отдельного компонента, который надо дальше сравнивать с FLiC-задержкой. "
            "Но для публикационного вывода нужен следующий слой: полноценная reverberation-модель диска и ошибки лагов.\n"
        )
    else:
        verdict.append(
            "На этом грубом тесте V-кандидат может быть совместим с простой дисковой задержкой. "
            "Нужен более строгий тест перед FLiC-интерпретацией.\n"
        )

    (out_dir / "disk_control_report_v4_1.md").write_text("".join(verdict), encoding="utf-8")

    print("[SAVED]", out_dir / "disk_control_best_lags_v4_1.csv")
    print("[SAVED]", out_dir / "disk_control_summary_v4_1.csv")
    print("[SAVED]", out_dir / "disk_control_report_v4_1.md")
    print("\n".join(verdict))


if __name__ == "__main__":
    main()
