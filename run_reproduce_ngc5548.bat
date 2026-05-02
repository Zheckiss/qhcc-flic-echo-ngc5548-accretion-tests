@echo off
REM Reproduce the core NGC 5548 FLiC accretion pipeline.
REM Existing downloaded files in run_v4\raw are reused unless you delete them or use --force-download.

python qhcc_ngc5548_run_v4.py ^
  --work-dir run_v4 ^
  --download ^
  --prepare ^
  --diagnose ^
  --baseline ^
  --rank

python qhcc_ngc5548_run_v4.py ^
  --work-dir run_v4 ^
  --null ^
  --null-pair hst_cos_1367.csv:opt_I_daily.csv ^
  --null-pair hst_cos_1367.csv:opt_R_daily.csv ^
  --n-null 1000 ^
  --null-mode shift ^
  --null-mode ou ^
  --null-mode false_lambda

python qhcc_ngc5548_flic_layer_test_v4_4.py ^
  --ranked run_v4\results\flic_candidates_ranked_v4.csv ^
  --baseline run_v4\results\baseline_summary_v4.csv ^
  --delays run_v4\results\ngc5548_flic_delays_v4.csv ^
  --null-pvalues run_v4\results\null_pvalues_combined_v4.csv ^
  --out-dir run_v4_4\results ^
  --n-random-lines 100000

python qhcc_ngc5548_disk_shape_test_v4_5.py ^
  --ranked run_v4\results\flic_candidates_ranked_v4.csv ^
  --delays run_v4\results\ngc5548_flic_delays_v4.csv ^
  --out-dir run_v4_5\results ^
  --min-local-z 2.5 ^
  --n-null 100000

echo Done. See run_v4_4\results and run_v4_5\results.
