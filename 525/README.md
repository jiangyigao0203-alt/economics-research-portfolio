# 525: Equity duration and monetary-policy shocks

This project studies heterogeneity in equity-return responses to monetary-policy and information shocks using implied equity duration and asset beta.

## Code map

| Files | Purpose |
| --- | --- |
| `build_compustat_monthly_panel.py` | Download Compustat fundamentals and build a monthly panel from quarterly inputs |
| `download_crsp_daily_prices.py`, `download_market_and_risk_free_returns.py` | Obtain matched CRSP prices and market/risk-free returns through WRDS |
| `decompose_monetary_policy_shocks.py` | Separate monetary-policy and information shocks using sign-restricted rotations of rate and equity surprises |
| `estimate_equity_duration.py` | Implied-duration calculation with configurable assumptions and outlier handling |
| `diagnose_equity_duration.py` | Duration diagnostics |
| `estimate_event_abnormal_returns.py` | Estimate event-window abnormal returns using market-model inputs |
| `estimate_asset_market_betas.py` | Estimate firm asset betas |
| `estimate_duration_shock_panel.py` | Merge shocks, event returns, and lagged duration; estimate panel specifications and export diagnostics |
| `test_duration_coefficient_trend.py` | Inspect duration-bin coefficient patterns |

Source files have descriptive names and module/function documentation for readers unfamiliar with the original workspace. [Source provenance](../SOURCE_MANIFEST.md) maps the old filenames to the published names. Earlier annual-data, alternate panel/duration, and ad hoc plotting/filtering scripts are omitted; see [selection notes](../FILE_SELECTION.md).

## Selected saved outputs

The [results folder](results/) contains coefficient estimates, model and sample statistics, and a second-stage trend test of duration-bin coefficients. The saved sample statistics report 965,551 firm-event observations, 13,890 firms, and 193 events, with firm fixed effects and HC1 covariance. These describe the saved run and have not been independently reproduced.

The trend test is an exploratory regression on estimated bin coefficients. Its reported p-values do not by themselves establish a structural or causal result or account for the full first-stage coefficient covariance.

## Reproduction guide and current gaps

1. Install the root requirements and change into `525`.
2. Configure dates and firm selections in the WRDS collection scripts. Run `build_compustat_monthly_panel.py` to create `compustat_monthly_panel_from_quarterly_2000_2024.csv`; then review `download_crsp_daily_prices.py` and `download_market_and_risk_free_returns.py`.
3. Obtain the shock inputs from their original source. `decompose_monetary_policy_shocks.py` expects `USMPD.csv` with `date_time`, `FF3`, and `SP500`. Its output is `jk_shocks_ff3_sp500.csv`; downstream filenames differ and must be aligned deliberately.
4. Review `estimate_equity_duration.py` and record its parameters. It currently writes `monthly_implied_duration_years.csv`, whereas `estimate_duration_shock_panel.py` expects `Implied_Duration_g=0.6.csv`. Do not infer the parameters of an old output from its filename alone.
5. `estimate_event_abnormal_returns.py` expects `shocks_fed_jk_t.csv` and writes `forecasted_returns_5day_after_fomc_updated.csv`. `estimate_duration_shock_panel.py` instead expects `shocks.csv` and `abnormal_return.csv`. Reconcile names, schemas, and event-window definitions before proceeding.
6. Review and run `estimate_asset_market_betas.py`, then `estimate_duration_shock_panel.py` with consistent shock, duration, beta, and abnormal-return files. Outputs go to `panel_outputs_model_selectable`.
7. Point `test_duration_coefficient_trend.py` to the chosen coefficient output; its current default path differ from `estimate_duration_shock_panel.py`'s output folder.

The saved aggregate tables in `results/` can be inspected without raw data. The full pipeline needs authorized inputs and configuration reconciliation; it has not been executed as part of packaging.
