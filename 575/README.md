# 575: Financial bubbles and volatility

This project studies explosive price episodes and their relationship with volatility across cryptocurrencies, ETFs, and exchange rates.

## Code map

| File | Purpose |
| --- | --- |
| `get_etfs.py` | Download ETF price histories |
| `BSADF.py` | Compute recursive backward supremum ADF statistics with configurable lags |
| `start.py` | Self-contained Bitcoin experiment with zero-lag interval-union detection, Monte Carlo calibration, and a plot |
| `arch2.py` | Estimate GARCH volatility and overlay bubble detections |
| `model.py` | Regress bubble-episode duration or intensity on volatility and market-type controls |
| `MLE1.py` | Build a panel with GARCH volatility features from precomputed BSADF outputs |
| `MLE2.py` | Multi-seed logistic classification and threshold experiments |
| `MLE3.py` | Time-split classifier control using a random-noise feature |

The two detection scripts serve different purposes: `BSADF.py` produces endpoint statistics; `start.py` implements an interval-union experiment. Their outputs are not interchangeable. Earlier standalone Monte Carlo, duplicate plotting, valuation-data, and summary helpers are omitted; see [selection notes](../FILE_SELECTION.md).

## Selected saved result

The [results folder](results/) contains an intensity-regression coefficient table and its model summary from local run `run_20260412_071631` (April 12, 2026). It uses 1,411 bubble episodes, pre-bubble volatility, market-type controls, and HC1 covariance; the archived R-squared is approximately 0.262. These are historical outputs, not newly reproduced estimates or causal evidence.

The saved configuration records intensity as the dependent variable, logged prices, annualized volatility, market-type controls, no year dummies, and a minimum episode duration of five observations. Those options agree with the packaged `model.py` settings; the exact source version and source-data snapshot used in that historical run are not established.

## Reproduction guide and current gaps

1. Install the root requirements and change into `575`.
2. `get_etfs.py` writes ETF data into `dataset/`. Crypto and exchange-rate inputs must be supplied separately from their original sources; source citations and redistribution permissions remain to be completed.
3. For `BSADF.py`, supply `daily_pe.csv` in its expected schema or explicitly configure another input. The default output name contains `1lag`, but `ADF_LAGS` is currently 2; use the setting rather than the filename to identify the specification. This script alone does not calibrate critical values or produce final bubble labels.
4. `start.py` expects `cryptos/btc-usd-max.csv`. Review `prepare_daily_data`, the log-price setting, Monte Carlo settings, and CPU usage. It creates a timestamped output folder and may require substantial runtime. Its interval-union labels are not the `is_bubble` labels expected by the downstream BSADF-based analyses.
5. For `model.py` and `MLE1.py`, configure the input folders `ETFs_BSADF`, `Exchange_Rate_BSADF`, and `cryptos_BSADF`. These are not bundled. CSVs need `date`, `price`, and `is_bubble`; current configuration treats price as already logged. The calibration and label-construction procedure for those historical files must be established before full replication.
6. `MLE1.py` builds a panel under `dataset_build_outputs`, whereas `MLE2.py` and `MLE3.py` currently read `dataset_build_outputs/111/panel_dataset_full.csv`. Point the classifiers to the actual generated panel.
7. `MLE2.py` is a random-split stability experiment. `MLE3.py` is a time-split random-noise control, not the main predictive model. Examine temporal leakage, threshold selection, and class balance before interpreting predictive performance.
8. `arch2.py` expects a separately prepared `detection.csv` with `date`, `price`, `bsadf`, `critical_value`, and `is_bubble`; the current scripts do not automatically generate that exact merged file.

The repository is a selected code sample with documented manual steps. The historical empirical pipeline has not been rerun.
