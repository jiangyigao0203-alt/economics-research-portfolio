# Empirical Results

We report bubble identification, episode characteristics, and classification performance separately. The tables below summarize the accompanying CSV, JSON, and statistical-output files.

## GSADF identification

| Market category | Series | Rejections | Rejection rate | Mean GSADF |
| --- | ---: | ---: | ---: | ---: |
| Cryptocurrencies | 50 | 33 | 0.660 | 4.442 |
| ETFs | 50 | 39 | 0.780 | 3.512 |
| Exchange rates | 22 | 21 | 0.955 | 7.948 |

[Market summary](gsadf_market_summary.csv) · Per-series tests: [crypto](gsadf_crypto.csv), [ETFs](gsadf_etfs.csv), [exchange rates](gsadf_exchange_rates.csv).

The rejection count is the number of series whose global statistic exceeds its Monte Carlo critical value. A global non-rejection does not exclude every local BSADF episode.

## BSADF and volatility figures

Each figure displays logged prices, BSADF statistics against critical values, and fitted GARCH volatility. Labelled bubble intervals are highlighted.

| Cryptocurrencies | ETFs | Exchange rates |
| --- | --- | --- |
| [Bitcoin](figures/bitcoin_bsadf_garch.png) | [SPY](figures/spy_bsadf_garch.png) | [Sweden](figures/sweden_bsadf_garch.png) |
| [Ethereum](figures/ethereum_bsadf_garch.png) | [VTI](figures/vti_bsadf_garch.png) | [South Africa](figures/south_africa_bsadf_garch.png) |

## Duration and intensity regressions

Both regressions contain 1,411 episodes and use HC1 covariance, pre-bubble volatility, and market-type indicators. ETFs are the reference category.

| Regressor | Duration coefficient | Duration p-value | Intensity coefficient | Intensity p-value |
| --- | ---: | ---: | ---: | ---: |
| Intercept | 34.6338 | <0.001 | 0.0371 | 0.064 |
| Pre-bubble volatility | 1.1783 | 0.758 | 0.1658 | 0.001 |
| Exchange-rate indicator | 23.4309 | 0.010 | 0.0050 | 0.754 |
| Cryptocurrency indicator | -6.5325 | 0.268 | 0.1865 | <0.001 |
| R-squared | 0.0077 | | 0.2615 | |

Duration is measured in observations; intensity is the within-episode log-price range. The volatility association is stronger for intensity than for duration in these specifications.

- Duration: [coefficients](duration_regression_coefficients.csv), [complete model output](duration_regression_summary.txt).
- Intensity: [coefficients](intensity_regression_coefficients.csv), [complete model output](intensity_regression_summary.txt).

## Logit classification

### Chronological holdout

The earlier 70% of each series supplies 425,104 training observations; the later 30% supplies 182,269 test observations. The volatility-dependent threshold is selected on training F1.

| Sample | ROC AUC | F1 |
| --- | ---: | ---: |
| Training | 0.6971 | 0.2847 |
| Testing | 0.3830 | 0.0932 |

[All metrics](time_holdout_metrics.json) · [Logit coefficients](time_holdout_coefficients.csv) · [Threshold parameters](time_holdout_threshold.json) · [Fit history](time_holdout_fit_history.csv).

The weak chronological test performance contrasts with the training fit. The [constant-threshold comparison](time_holdout_constant_threshold_metrics.json) yields a similar test AUC and F1. The [noise-control result](noise_control_metrics.csv) is a separate diagnostic specification, not the volatility model.

### Random-market holdouts

We hold out approximately 30% of series within each market category and repeat the split across 50 seeds. Whole series, rather than observations, are assigned to the test group.

| Metric | Training mean | Training SD | Testing mean | Testing SD |
| --- | ---: | ---: | ---: | ---: |
| ROC AUC | 0.5981 | 0.0141 | 0.6091 | 0.0430 |
| F1 | 0.2285 | 0.0178 | 0.2189 | 0.0491 |

[All 50 runs](series_holdout_50_runs.csv) · [Summary statistics](series_holdout_summary.csv).

The random-market and chronological designs evaluate different forms of generalization. Better average performance across held-out series does not resolve the poor chronological holdout result.

## Sample composition

[Overall feature-panel summary](sample_summary.json) · [By market type](sample_by_market_type.csv) · [By observation frequency](sample_by_frequency.csv).
