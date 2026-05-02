# QHCC/FLiC accretion echo pilot for NGC 5548 — v4

Эта версия исправляет главное слабое место v3: целевые FLiC-задержки больше не зашиты в коде. Они вычисляются из `config_ngc5548_v4.json`: массы центральной ЧД, красного смещения и списка ветвей.

## Главные изменения v4

1. Основные научные ветви только две:
   - `one_way_2over3`: \(\alpha=2/3\);
   - `two_way_4over3`: \(\alpha=4/3\).
2. Ветка `alpha=3/4` убрана из научного поиска. Если она понадобится как диагностический тест, её можно явно добавить в `branches` в конфиге.
3. `baseline`, `lag-pair` и `null` используют один и тот же список целевых задержек.
4. Добавлен режим `false_lambda`: проверяет, возникают ли похожие пики вне заранее заданной области \(\lambda\approx1\).
5. Добавлен агрегатор `--rank`, который собирает `baseline_summary_v4.csv` в `flic_candidates_ranked_v4.csv`.

## Файлы

- `qhcc_ngc5548_core_v3.py` — слой загрузки и подготовки данных, используется как зависимость.
- `qhcc_ngc5548_core_v4.py` — новая логика расчёта целей, baseline, null и ranking.
- `qhcc_ngc5548_run_v4.py` — единая точка запуска.
- `config_ngc5548_v4.json` — конфиг массы, красного смещения, ветвей и списка кривых отклика.

## Базовый запуск

```bash
python qhcc_ngc5548_run_v4.py --work-dir run_v4 --download --prepare --diagnose --baseline --rank
```

Если сырые CDS-файлы уже скачаны в `run_v4/raw`, можно запускать без `--download`:

```bash
python qhcc_ngc5548_run_v4.py --work-dir run_v4 --prepare --diagnose --baseline --rank
```

## Нулевые тесты для выбранных пар

```bash
python qhcc_ngc5548_run_v4.py --work-dir run_v4 --null \
  --null-pair hst_cos_1367.csv:opt_I_daily.csv \
  --null-pair hst_cos_1367.csv:opt_R_daily.csv \
  --n-null 1000 \
  --null-mode shift \
  --null-mode ou \
  --null-mode false_lambda
```

## Как читать результаты

Главные выходные файлы:

- `results/ngc5548_flic_delays_v4.csv` — ожидаемые FLiC-задержки из массы и красного смещения.
- `results/baseline_summary_v4.csv` — результаты по всем парам драйвер/отклик.
- `results/flic_candidates_ranked_v4.csv` — ранжированный список кандидатов.
- `results/null_*_v4.pvalues.csv` — эмпирические p-values по нулевым режимам.

Ключевые колонки:

- `tau_obs_days_exact` — предсказанная FLiC-задержка в системе наблюдателя.
- `tau_at_target_grid_days` — ближайший лаг сетки к предсказанной задержке.
- `tau_peak_days` — максимум DCF-score в заранее заданном окне вокруг цели.
- `lambda_peak` — отношение `tau_peak_days / tau_target_days`.
- `dcf_at_target` — DCF-score в точке строгой цели.
- `dcf_peak` — максимум DCF-score в окне.
- `local_z_target` — локальная z-оценка строгого попадания.
- `local_z_peak` — локальная z-оценка максимума в окне.

Для строгой интерпретации главный тест — `target`. `peak` является поддерживающим и требует поправки за поиск максимума внутри окна.
