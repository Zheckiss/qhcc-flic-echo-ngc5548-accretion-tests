# NGC 5548 disk-only shape vs FLiC-dominated lag layer v4.5

This test does not perform a new echo search. It takes the selected one-way DCF peaks from v4
and asks whether their lag-vs-wavelength shape looks like disk-only reverberation or like
a fixed FLiC delay plus a small wavelength-dependent correction.

## Selection

- Target branch: `one_way_2over3`
- FLiC delay: **12.796 h**
- Lambda window: `0.75` to `1.3`
- Minimum local peak z: `2.5`
- Unique physical bands: `True`
- Selected points: **10**

## Model comparison

| model                      |   rms_hours |   A_hours |      beta |   slope_b_hours_per_log_lambda |   offset_hours |   pivot_A |
|:---------------------------|------------:|----------:|----------:|-------------------------------:|---------------:|----------:|
| disk_only_beta_4over3      |     6.72506 |   1.98311 |   1.33333 |                       nan      |     nan        |    nan    |
| disk_only_free_beta        |     4.65129 |  26.4144  |   0.3     |                       nan      |     nan        |    nan    |
| flic_constant              |     2.16401 | nan       | nan       |                         0      |       0        |    nan    |
| flic_log_slope             |     1.42923 | nan       | nan       |                         3.2145 |       0        |   4294.38 |
| flic_log_slope_with_offset |     1.356   | nan       | nan       |                         3.2145 |       0.451645 |   4294.38 |

## Key result


## Null checks

- Wavelength-permutation p(RMS as good as real): **0.0159**
- Wavelength-permutation p(RMS as good and slope at least as positive): **0.0064**
- Random-center p(RMS as good as fixed FLiC center): **0.0353**

## Interpretation

A pure disk-only lag curve is expected to be primarily wavelength-dependent. The FLiC-dominated
model instead predicts a nearly horizontal layer near the fixed FLiC delay, with a small
positive wavelength-dependent correction. This script quantifies that shape distinction.

The result should be read as a shape diagnostic, not as a new independent detection claim.