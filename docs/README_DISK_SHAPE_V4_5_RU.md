# QHCC NGC 5548 disk-shape test v4.5

## Что это

`v4.5` не ищет эхо заново. Он берёт уже найденные DCF-пики из `flic_candidates_ranked_v4.csv` и проверяет форму графика:

```text
чистый диск
vs
FLiC-задержка + малая поправка по длине волны
```

Главный вопрос:

```text
чистый диск должен дать наклонную кривую,
а наблюдаемые точки образуют почти горизонтальный слой около 12.8 ч?
```

## Быстрый запуск

Положи скрипт рядом с файлами:

```text
flic_candidates_ranked_v4.csv
ngc5548_flic_delays_v4.csv
```

и запускай:

```bat
python qhcc_ngc5548_disk_shape_test_v4_5.py ^
  --ranked flic_candidates_ranked_v4.csv ^
  --delays ngc5548_flic_delays_v4.csv ^
  --out-dir run_v4_5\results ^
  --min-local-z 2.5 ^
  --n-null 100000
```

## Что он считает

Модели:

1. Чистый диск с фиксированным beta=4/3:

```text
tau = A [(lambda/1367)^4/3 - 1]
```

2. Чистый диск со свободным beta:

```text
tau = A [(lambda/1367)^beta - 1]
```

3. Чистый FLiC-слой:

```text
tau = Delta_t_FLIC
```

4. FLiC + малый наклон:

```text
tau = Delta_t_FLIC + b log(lambda/lambda_pivot)
```

## Главные выходы

```text
flic_disk_shape_points_v4_5.csv
flic_disk_shape_model_comparison_v4_5.csv
flic_disk_shape_nulls_v4_5.csv
flic_disk_shape_report_v4_5.md
flic_disk_shape_comparison_v4_5.png
flic_disk_shape_residuals_v4_5.png
```

## Как читать результат

Главный файл:

```text
flic_disk_shape_model_comparison_v4_5.csv
```

Смотри:

```text
rms_hours
```

Если:

```text
RMS(FLiC + малый наклон) << RMS(чистый диск)
```

значит наблюдаемая форма графика больше похожа на FLiC-доминированный слой, чем на чистую дисковую реверберацию.

## Нули

Скрипт делает два простых нуля:

1. Перестановка длин волн между найденными лагами:
   проверяет, является ли правильный наклон по длине волны случайным.

2. Случайный центр горизонтального слоя:
   проверяет, насколько специальна именно заранее заданная FLiC-шкала 12.8 ч.

Это shape-test, не новая независимая детекция.
