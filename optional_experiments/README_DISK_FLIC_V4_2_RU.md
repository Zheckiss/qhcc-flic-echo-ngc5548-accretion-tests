# NGC 5548 disk-only vs disk+FLiC control v4.2

Этот пакет добавляет следующий контроль после `v4`: проверку, может ли главный V-кандидат быть обычной гладкой задержкой аккреционного диска.

## Что проверяется

Главный кандидат:

```text
HST/COS 1367 Å -> opt_V_daily.csv
ветвь FLiC: two_way_4over3
```

Сравниваются три варианта:

1. гладкий диск с фиксированным показателем `beta = 4/3`;
2. гладкий диск со свободным `beta`;
3. фиксированная FLiC-задержка `two_way_4over3`.

Модель диска:

```text
tau(lambda) = A * [ (lambda / 1367 Å)^beta - 1 ]
```

Это не полная физическая модель диска, а первый строгий sanity-check: если V-пик является обычным дисковым лагом, он должен лечь на плавную зависимость задержки от длины волны.

## Команда запуска

```bat
python qhcc_ngc5548_disk_flic_compare_v4_2.py ^
  --ranked run_v4\results\flic_candidates_ranked_v4.csv ^
  --baseline run_v4\results\baseline_summary_v4.csv ^
  --delays run_v4\results\ngc5548_flic_delays_v4.csv ^
  --null-pvalues run_v4\results\null_pvalues_combined_v4.csv ^
  --out-dir run_v4\results ^
  --target-response opt_V_daily.csv ^
  --flic-target-name two_way_4over3
```

## Выходные файлы

```text
disk_flic_fit_points_v4_2.csv
disk_flic_model_summary_v4_2.csv
disk_flic_model_predictions_v4_2.csv
disk_flic_target_nulls_v4_2.csv  (если найдены нулевые p-значения для целевого канала)
disk_flic_compare_report_v4_2.md
disk_flic_lag_vs_wavelength_v4_2.png
disk_flic_target_residual_v4_2.png
```

## Как читать результат

Главный файл:

```text
disk_flic_model_summary_v4_2.csv
```

Смотри строки:

```text
disk_only_beta_fixed_4over3
disk_only_beta_free
fixed_flic_component
```

Если остаток V относительно `fixed_flic_component` намного меньше, чем относительно `disk_only_*`, значит V-кандидат лучше совпадает с фиксированной FLiC-задержкой, чем с гладкой цветовой задержкой диска.

## Важно

Этот v4.2 не заменяет полную модель аккреционного диска. Следующая версия должна добавить широкую функцию отклика, фотометрические ошибки, FR/RSS или DRW/CARMA симуляции и сравнение disk-only против disk+FLiC на уровне light curve likelihood.
