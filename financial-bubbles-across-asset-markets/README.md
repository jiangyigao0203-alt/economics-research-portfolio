# Identifying Financial Bubbles Across Heterogeneous Asset Markets: Evidence from Cryptocurrencies, ETFs, and Exchange Rates

**Yigao Jiang and Matthew Lam — April 2026**

We compare explosive price behavior across cryptocurrencies, ETFs, and exchange rates. We use GSADF to test for whole-series explosiveness, BSADF to identify dated episodes, GARCH volatility to characterize duration and intensity, and logit models to evaluate bubble classification under two sample-splitting designs.

## Empirical results

The [results guide](results/README.md) presents the GSADF summary, duration and intensity regressions, chronological-holdout metrics, and all 50 random-market-holdout runs. It also links representative Bitcoin, Ethereum, ETF, and exchange-rate identification figures.

## Methods and code

| Component | Source | Specification |
| --- | --- | --- |
| Whole-series GSADF tests | [cpp/multi_market_gsadf.cpp](cpp/multi_market_gsadf.cpp) | Fixed one-lag ADF regressions; 500 null simulations; one result per series |
| Combined BSADF and GSADF | [cpp/multi_market_bsadf_gsadf.cpp](cpp/multi_market_bsadf_gsadf.cpp) | Configurable lag list; endpoint critical values and global tests |
| Large zero-lag calibration | [cpp/bsadf_monte_carlo_lag0.cpp](cpp/bsadf_monte_carlo_lag0.cpp) | 10,000 null paths at the configured sample length |
| Market-category rejection summary | `summarize_gsadf_tests.py` | Aggregate series-level GSADF results |
| ETF acquisition | `download_etf_prices.py` | Daily Yahoo Finance prices |
| Standalone BSADF calculation | `compute_bsadf.py` | Recursive endpoint statistics without calibration |
| Interval-union experiment | `detect_bubbles_interval_union.py` | Zero-lag maximum-over-intervals calibration |
| GARCH figures | `plot_bubbles_and_garch_volatility.py` | Price, test statistic, critical value, and conditional volatility |
| Duration and intensity | `regress_bubble_episode_characteristics.py` | Episode-level OLS with HC1 covariance |
| Classification panel | `build_bubble_volatility_panel.py` | Labels, lagged GARCH volatility, and market indicators |
| Chronological evaluation | `evaluate_classifier_time_holdout.py` | Earlier 70% of each series for training; later 30% for testing |
| Random-market evaluation | `evaluate_classifier_series_holdout.py` | Hold out whole series within market type; repeat 50 seeds |
| Noise control | `evaluate_classifier_noise_control.py` | Chronological evaluation with seeded noise replacing volatility |

## Statistical design

The ADF regression includes an intercept, a lagged price level, and the selected lagged differences. GSADF maximizes over admissible starting and ending dates; BSADF maximizes over starting dates for each ending date. We obtain empirical 95th-percentile critical values from random-walk simulations. Local BSADF labels are retained independently of the global GSADF rejection decision, with a minimum episode length of five observations.

Episode duration is the number of observations in a contiguous labelled run. Intensity is the range of logged prices within that episode. The regressions use volatility immediately before the episode and market-type indicators, with ETFs as the reference category.

The logit model uses lagged GARCH volatility and exchange-rate/crypto indicators. Continuous features are standardized using training-sample moments. A logistic volatility-dependent classification threshold is selected to maximize training F1 and then applied to the test sample. Chronological and random-market splits answer different questions and are reported separately.

## Running the analysis

Build the native programs using the [C++ guide](cpp/README.md). Price CSVs belong under `data/markets/ETFS`, `data/markets/ExchangeRate`, and `data/markets/crypto`. The combined program exports logged `price`, `date`, and `is_bubble`; these are the inputs to the episode and feature-building scripts. Select one lag specification per analysis to avoid counting the same series more than once.

Set `INPUT_FOLDERS` in the feature and episode scripts to the selected BSADF files. Set the classifier scripts' `INPUT_PANEL_FILE` to the resulting panel. Configuration blocks retain the independent experiments' input/output filenames; the standalone BSADF and interval-union outputs are not substitutes for the combined program's calibrated labels.

## Interpretation

The identification sample contains 50 cryptocurrency series, 50 ETFs, and 22 exchange-rate series. The feature panel contains 607,373 observations and 1,411 bubble starts. Coverage varies by asset, and short histories limit identification and volatility estimation. Bubble labels and minimum-duration filtering are retrospective. GARCH feature construction and label construction must be considered when interpreting classification as forecasting performance. The chronological test AUC is below 0.5 even though random-market holdouts perform better on average.
