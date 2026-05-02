# QHCC/FLiC NGC 5548 flare-pair search v5.3

Версия v5.3 — это all-band event-based тест аккреционных вспышек без привилегированного диапазона.

Главное отличие от v5.2:

- добавлен `--unique-physical-bands`, чтобы не считать дубли одной физической кривой как независимые каналы;
- добавлен `--exclude`, чтобы вручную исключать конкретные файлы или маски;
- добавлен `curve_inventory_v5_3.csv`, где видно, какие кривые использованы, а какие пропущены;
- сохранено распараллеливание нулевых тестов через `--workers`.

## Зачем это нужно

В предыдущем прогоне одновременно присутствовали:

```text
hst_cos_1367.csv
hst_cos_1367_paper1.csv
```

Они могут представлять почти один и тот же физический диапазон 1367 Å. Для финального анализа надо проверить, сохраняется ли результат без такого дублирования.

## Рекомендуемый быстрый прогон без дублей

Запускать из папки, где лежит скрипт и папка `curves`:

```bat
python qhcc_ngc5548_flare_pair_search_v5_3.py ^
  --curves-dir curves ^
  --out-dir run_v5_3\results_fast_nodup ^
  --min-z 2.0 ^
  --tolerance-hours 1.5 ^
  --n-null 200 ^
  --workers 20 ^
  --cross-band-only ^
  --unique-physical-bands
```

## Эквивалентный ручной режим

Если хочешь исключить только конкретный дубль:

```bat
python qhcc_ngc5548_flare_pair_search_v5_3.py ^
  --curves-dir curves ^
  --out-dir run_v5_3\results_fast_exclude_paper1 ^
  --min-z 2.0 ^
  --tolerance-hours 1.5 ^
  --n-null 200 ^
  --workers 20 ^
  --cross-band-only ^
  --exclude hst_cos_1367_paper1.csv
```

## Финальный прогон

После быстрого теста:

```bat
python qhcc_ngc5548_flare_pair_search_v5_3.py ^
  --curves-dir curves ^
  --out-dir run_v5_3\results_final_nodup ^
  --min-z 2.0 ^
  --tolerance-hours 1.5 ^
  --n-null 1000 ^
  --workers 20 ^
  --cross-band-only ^
  --unique-physical-bands
```

## Какие файлы прислать обратно

Минимум:

```text
flare_pair_summary_v5.csv
flare_pair_report_v5.md
flare_pairs_flic_hits_v5.csv
flare_pairs_flic_hits_ranked_v5.csv
flare_catalog_v5.csv
curve_inventory_v5_3.csv
```

Желательно также:

```text
flare_pair_null_distribution_v5.csv
```

## Как читать результат

Главная проверка v5.3:

- результат v5.2 был ли раздут дублями;
- остаётся ли избыток 1-way FLiC-пар после режима `--unique-physical-bands`;
- меняются ли p-values относительно ложных лагов, случайных времён и сдвигов диапазонов.

Если 1-way остаётся заметным после удаления дублей, это укрепляет event-based аккреционный тест QHCC/FLiC.
