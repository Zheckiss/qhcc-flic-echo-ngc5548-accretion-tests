# NGC 5548 FLiC layer organisation test v4.4

This report formalises the visual lag--wavelength structure seen in the v4.2
figure. It does **not** perform a new echo search. It takes the already ranked
v4 candidates and tests whether the selected lags are organised around the fixed
FLiC scales.

## Fixed FLiC scales

- One-way FLiC branch: **12.796 h**
- Two-way FLiC branch: **25.593 h**

## One-way layer result

- Number of one-way layer points: **8**
- RMS around the one-way FLiC branch: **1.881 h**
- Mean absolute deviation around the one-way branch: **1.718 h**
- Standard deviation of one-way residuals: **2.000 h**
- Mean residual for short-wavelength channels: **-1.276 h**
- Mean residual for long-wavelength channels: **1.604 h**

Interpretation: short-wavelength channels lie below the one-way FLiC scale on
average, while long-wavelength optical channels lie above it. This turns the
visual pattern into a measurable layer structure.

## Comparison with disk-only organisation

- RMS around one-way FLiC: **1.881 h**
- RMS around disk-only curve with fixed beta=4/3: **6.539 h**
- RMS around best disk-only curve with free beta: **4.755 h**
- Best disk-only beta: **0.300**

In this diagnostic, the selected one-way points are more tightly organised around
the fixed one-way FLiC line than around the simple disk-only lag curve.

## Random horizontal line test

Random horizontal lag lines were sampled between
5.0 h and 30.0 h.

- Median random-line RMS: **6.523 h**
- Probability of a random line being as good as or better than the FLiC one-way line:
  **p = 0.0155**
- Best random line in the scan: **12.600 h**
- RMS of the best random line: **1.871 h**

This is not a full global false-alarm probability. It is a structural check:
it asks whether an arbitrary horizontal lag scale would organise the one-way
points as well as the pre-fixed FLiC one-way branch.

## Two-way branch

- Main two-way candidate: opt V at 25.44 h, two-way FLiC expectation 25.59 h, delta -0.15 h, lambda_two_way=0.994.

The two-way branch is therefore treated as the cleanest completed FLiC candidate,
while the one-way branch is treated as a structured early-response layer.

## Files produced

- `flic_layer_points_v4_4.csv`
- `flic_one_way_layer_summary_v4_4.csv`
- `flic_two_way_summary_v4_4.csv`
- `flic_layer_rms_comparison_v4_4.csv`
- `flic_lag_vs_wavelength_v4_4.png`
- `flic_residuals_one_way_v4_4.png`
- `flic_layer_rms_comparison_v4_4.png`
