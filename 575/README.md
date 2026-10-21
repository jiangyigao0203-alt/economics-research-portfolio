# 575: Financial bubbles and volatility

This project combines C++ simulation and bubble-test programs with Python volatility estimation, episode regressions, and classifier experiments.

## Workflow and code map

| Stage | File | Purpose |
| --- | --- | --- |
| Price acquisition | `download_etf_prices.py` | Download daily ETF histories; crypto and exchange-rate inputs are supplied separately |
| Calibrated multi-market tests | [cpp/multi_market_bsadf_gsadf.cpp](cpp/multi_market_bsadf_gsadf.cpp) | Simulate null distributions, compute BSADF paths and GSADF tests, and export filtered bubble labels for each configured lag |
| Dedicated large calibration | [cpp/bsadf_monte_carlo_lag0.cpp](cpp/bsadf_monte_carlo_lag0.cpp) | Simulate endpoint-specific zero-lag BSADF critical values; defaults use 10,000 null paths |
| Standalone statistic | `compute_bsadf.py` | Compute endpoint BSADF statistics without calibration or final labels |
| Alternative detection experiment | `detect_bubbles_interval_union.py` | Calibrate a maximum-over-intervals zero-lag statistic and flag the union of exceeding intervals |
| Volatility plot | `plot_bubbles_and_garch_volatility.py` | Estimate GARCH(1,1) variance and plot it alongside bubble labels |
| Episode analysis | `regress_bubble_episode_characteristics.py` | Regress episode duration or intensity on volatility and market-type controls |
| Classifier features | `build_bubble_volatility_panel.py` | Construct a dated panel of labels and GARCH volatility features |
| Series holdout | `evaluate_classifier_series_holdout.py` | Repeat logistic classification while holding out entire series within market type |
| Noise control | `evaluate_classifier_noise_control.py` | Split each series chronologically and replace volatility with seeded random noise |

See the [C++ build and data-interface guide](cpp/README.md) for detailed simulation instructions, expected schemas, calibration settings, and output meanings. The three detection approaches have different thresholds and labels; they are not interchangeable.

## Running the analysis

1. Install the root Python requirements. For the simulation programs, build the C++ sources using their CMake instructions.
2. Supply positive-price CSVs under `575/data/markets/ETFS`, `ExchangeRate`, and `crypto`. The combined C++ program reads the first two columns as date and price, logs prices, and processes its configured lag list. Execute it from `575/cpp` so its relative defaults resolve correctly.
3. Select exactly one lag specification per Python analysis. Configure separate input folders in the episode-regression and feature-builder scripts with the matching BSADF files. These files contain the required `date`, logged `price`, and `is_bubble`. Do not duplicate a series by mixing lag variants.
4. The feature builder creates a timestamped panel under `dataset_build_outputs`. Point both classifier scripts' `INPUT_PANEL_FILE` at the actual generated panel; their retained defaults still contain the historical `111` directory.
5. The series-holdout experiment tests transfer across market series, not a chronological forecasting split. The chronological noise experiment is a diagnostic control. Neither design establishes real-time predictability of retrospective bubble labels; examine label construction, feature estimation, and temporal leakage before making a predictive claim.

Other retained defaults are documented in each script. `compute_bsadf.py` expects `daily_pe.csv`, and its output filename says `1lag` even though `ADF_LAGS` is 2; use the configured setting to identify the specification. The interval-union experiment expects `cryptos/btc-usd-max.csv`. The volatility plot expects a merged `detection.csv` with `date`, `price`, `bsadf`, `critical_value`, and `is_bubble`. Renaming and documentation have not silently reconciled these separate experiment inputs.

## Selected historical result

The [results folder](results/) contains the intensity-regression coefficients and model summary from local run `run_20260412_071631` (April 12, 2026): 1,411 bubble episodes, pre-bubble volatility, market-type controls, HC1 covariance, and R-squared approximately 0.262. They are historical outputs rather than newly replicated estimates or causal evidence. The exact source version and source-data snapshot underlying that run remain unverified, including whether the newly supplied C++ source generated its labels.

Raw datasets, compiled executables, and large generated outputs are excluded. Source citations, input access, and full empirical replication still require work. [Selection notes](../FILE_SELECTION.md) explain which files were kept and omitted.
