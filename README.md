# qhcc-flic-echo-ngc5548-accretion-tests

Reproducible QHCC/FLiC accretion-lag framework for NGC 5548.

Author: Evgeniy Malkov  
GitHub profile: https://github.com/Zheckiss/  
Related work: **QHCC Practical Note II: FLiC-Dominated Accretion Lag Layer in NGC 5548**

## What this repository tests

This repository reproduces the NGC 5548 accretion-side FLiC analysis.

The tested observer-frame FLiC delay is

```text
Delta t_obs(M; alpha, z) = (1 + z) alpha (r_S / c) ln(r_S / l_P)
```

with the two main branches:

```text
alpha = 2/3   one-way
alpha = 4/3   two-way
```

For NGC 5548, using `M = 6.5e7 Msun` and `z = 0.017175`, the fixed targets are approximately:

```text
one-way: 12.796 h
two-way: 25.593 h
```

The final Note II interpretation is:

```text
NGC 5548 shows a FLiC-dominated accretion lag layer:
a fixed QHCC/FLiC delay plus a small wavelength-dependent disk correction,
rather than a pure disk-only lag curve.
```

## Repository layout

```text
qhcc_ngc5548_core_v3.py
qhcc_ngc5548_core_v4.py
qhcc_ngc5548_run_v4.py
qhcc_ngc5548_flic_layer_test_v4_4.py
qhcc_ngc5548_disk_shape_test_v4_5.py
config_ngc5548_v4.json

run_reproduce_ngc5548.bat
run_reproduce_ngc5548.sh

docs/
paper_results/
optional_experiments/
```

Core scripts:

- `qhcc_ngc5548_run_v4.py`  
  Builds the v4 DCF baseline/ranking and FLiC delay targets.

- `qhcc_ngc5548_flic_layer_test_v4_4.py`  
  Tests the one-way FLiC lag layer around the fixed 12.796 h target.

- `qhcc_ngc5548_disk_shape_test_v4_5.py`  
  Tests pure disk-only shape versus FLiC-dominated layer plus a small wavelength correction.

The `optional_experiments/` directory contains later flare-pair/stacking experiments and earlier disk-control scripts. These are not the central evidence in Note II.

## Install

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Minimal reproduction order

### 1. Build v4 baseline and ranked candidates

Windows:

```bat
python qhcc_ngc5548_run_v4.py ^
  --work-dir run_v4 ^
  --download ^
  --prepare ^
  --diagnose ^
  --baseline ^
  --rank
```

Linux/macOS:

```bash
python qhcc_ngc5548_run_v4.py \
  --work-dir run_v4 \
  --download \
  --prepare \
  --diagnose \
  --baseline \
  --rank
```

If raw data are already present in `run_v4/raw`, rerun without `--download`:

```bash
python qhcc_ngc5548_run_v4.py --work-dir run_v4 --prepare --diagnose --baseline --rank
```

Main outputs:

```text
run_v4/results/ngc5548_flic_delays_v4.csv
run_v4/results/baseline_summary_v4.csv
run_v4/results/flic_candidates_ranked_v4.csv
```

### 2. Run selected v4 null tests

Windows:

```bat
python qhcc_ngc5548_run_v4.py ^
  --work-dir run_v4 ^
  --null ^
  --null-pair hst_cos_1367.csv:opt_I_daily.csv ^
  --null-pair hst_cos_1367.csv:opt_R_daily.csv ^
  --n-null 1000 ^
  --null-mode shift ^
  --null-mode ou ^
  --null-mode false_lambda
```

Main output:

```text
run_v4/results/null_pvalues_combined_v4.csv
```

### 3. Run v4.4 FLiC lag-layer diagnostic

```bat
python qhcc_ngc5548_flic_layer_test_v4_4.py ^
  --ranked run_v4esultslic_candidates_ranked_v4.csv ^
  --baseline run_v4esultsaseline_summary_v4.csv ^
  --delays run_v4esults
gc5548_flic_delays_v4.csv ^
  --null-pvalues run_v4esults
ull_pvalues_combined_v4.csv ^
  --out-dir run_v4_4esults ^
  --n-random-lines 100000
```

Main outputs:

```text
run_v4_4/results/flic_lag_vs_wavelength_v4_4.png
run_v4_4/results/flic_layer_test_report_v4_4.md
run_v4_4/results/flic_layer_rms_comparison_v4_4.csv
```

### 4. Run v4.5 disk-shape diagnostic

```bat
python qhcc_ngc5548_disk_shape_test_v4_5.py ^
  --ranked run_v4esultslic_candidates_ranked_v4.csv ^
  --delays run_v4esults
gc5548_flic_delays_v4.csv ^
  --out-dir run_v4_5esults ^
  --min-local-z 2.5 ^
  --n-null 100000
```

Main outputs:

```text
run_v4_5/results/flic_disk_shape_comparison_v4_5.png
run_v4_5/results/flic_disk_shape_report_v4_5.md
run_v4_5/results/flic_disk_shape_model_comparison_v4_5.csv
```

## One-command reproduction

Windows:

```bat
run_reproduce_ngc5548.bat
```

Linux/macOS:

```bash
bash run_reproduce_ngc5548.sh
```

## Expected final pattern

The final Note II result is not a single optical point. It is the shape of the lag--wavelength diagram:

- selected one-way lag peaks form a low-jitter layer around 12.796 h;
- the cleanest two-way optical point lies near 25.593 h;
- disk-only lag curves have substantially worse RMS than a FLiC-dominated layer plus a small wavelength correction.

Representative final figures and reports are included in `paper_results/`.

## Data policy

Raw public NGC 5548 reverberation-mapping files are not stored in this repository. The v4 runner downloads/prepares them locally when `--download` is used. Generated folders such as `run_v4/`, `run_v4_4/`, and `run_v4_5/` are excluded from Git.

## Citation

Suggested repository URL after upload:

```text
https://github.com/Zheckiss/qhcc-flic-echo-ngc5548-accretion-tests
```
