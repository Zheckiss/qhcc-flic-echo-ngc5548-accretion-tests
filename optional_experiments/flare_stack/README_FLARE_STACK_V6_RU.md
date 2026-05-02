# QHCC NGC 5548 flare-triggered stack v6

## Что это

`v6` — это all-band flare-triggered stacking test.

Он больше не выбирает один диапазон как физический драйвер.  
Методика такая:

1. берём все кривые блеска из `curves`;
2. в каждой ищем значимые вспышечные маркеры;
3. близкие по времени маркеры склеиваем в один уникальный якорь события;
4. вокруг каждого якоря ставим `t = 0`;
5. вырезаем окно световых кривых от `-10` до `+80` часов;
6. складываем все окна;
7. проверяем, появляется ли средний горб на FLiC-временах:
   - `12.796 ч` для one-way `2/3`;
   - `25.593 ч` для two-way `4/3`.

Это ближе к физике аккреционного FLiC-эхо, чем прямой перебор всех пар вспышек.

## Быстрый тест

Из папки, где лежит скрипт и папка `curves`:

```bat
python qhcc_ngc5548_flare_stack_v6.py ^
  --curves-dir curves ^
  --out-dir run_v6\results_fast ^
  --min-z 2.0 ^
  --unique-physical-bands ^
  --n-null 100 ^
  --workers 20
```

## Рабочий прогон

```bat
python qhcc_ngc5548_flare_stack_v6.py ^
  --curves-dir curves ^
  --out-dir run_v6\results_final ^
  --min-z 2.0 ^
  --unique-physical-bands ^
  --n-null 1000 ^
  --workers 20
```

## Более строгий прогон

```bat
python qhcc_ngc5548_flare_stack_v6.py ^
  --curves-dir curves ^
  --out-dir run_v6\results_z25 ^
  --min-z 2.5 ^
  --unique-physical-bands ^
  --n-null 1000 ^
  --workers 20
```

## Главные выходные файлы

```text
curve_inventory_v6.csv
flare_marker_catalog_v6.csv
anchor_catalog_v6.csv
stack_profile_allbands_v6.csv
echo_window_stats_v6.csv
stack_null_distribution_v6.csv
flare_stack_report_v6.md
flare_stack_profile_v6.png
flare_stack_nulls_v6.png
```

## Как читать результат

Главный файл:

```text
echo_window_stats_v6.csv
```

Главные колонки:

```text
echo_mean_z
sideband_mean_z
echo_excess_z
p_excess_random_anchor
p_excess_false_lag
```

Смысл:

- `echo_excess_z` — насколько средний stack в FLiC-окне выше локальных боковых окон;
- `p_excess_random_anchor` — как часто случайные времена дают такой же или больший горб;
- `p_excess_false_lag` — как часто ложные лаги внутри того же stack дают такой же или больший горб.

## Что считать сильным результатом

Сильный результат:

```text
echo_excess_z > 0
p_excess_random_anchor small
p_excess_false_lag small
```

Особенно если это устойчиво при:

```text
min-z = 2.0 и 2.5
bin-hours = 1.0 и 2.0
echo-half-width-hours = 1.0 и 1.5
```

## Важное ограничение

`v6` не утверждает, что точность лагов субчасовая.  
Он проверяет когерентный средний отклик в заранее заданных FLiC-окнах.

Если эхо реально есть и связано со вспышками, случайный шум должен усредняться, а FLiC-компонент должен оставаться как горб около `12.8` и/или `25.6` часов.
