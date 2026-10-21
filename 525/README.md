# 525: Equity duration and monetary-policy shocks

This project studies heterogeneity in equity-return responses to monetary-policy and information shocks using implied equity duration and asset beta.

## Code map

| Files | Purpose |
| --- | --- |
| `getdatawrdsmonthly.py` | Download Compustat fundamentals and build a monthly panel from quarterly inputs |
| `get_daily_price.py`, `market_data.py` | Obtain matched CRSP prices and market/risk-free returns through WRDS |
| `Decomposition.py` | Separate monetary-policy and information shocks using sign-restricted rotations of rate and equity surprises |
| `impliedduration.py` | Implied-duration calculation with configurable assumptions and outlier handling |
| `diag.py` | Duration diagnostics |
| `CAMP.py` | Estimate event-window abnormal returns using market-model inputs |
| `risk_factor.py` | Estimate firm asset betas |
| `panel.py` | Merge shocks, event returns, and lagged duration; estimate panel specifications and export diagnostics |
| `increase_test.py` | Inspect duration-bin coefficient patterns |

Original filenames are retained. Earlier annual-data, alternate panel/duration, and ad hoc plotting/filtering scripts are omitted; see [the selection notes](../FILE_SELECTION.md).

## Selected saved outputs

The [results folder](results/) contains coefficient estimates, model and sample statistics, and a second-stage trend test of duration-bin coefficients. The saved sample statistics report 965,551 firm-event observations, 13,890 firms, and 193 events, with firm fixed effects and HC1 covariance. These describe the saved run and have not been independently reproduced.

The trend test is an exploratory regression on estimated bin coefficients. Its reported p-values do not by themselves establish a structural or causal result or account for the full first-stage coefficient covariance.

## Reproduction guide and current gaps

1. Install the root requirements and change into `525`.
2. Configure dates and firm selections in the WRDS collection scripts. Run `getdatawrdsmonthly.py` to create `compustat_monthly_panel_from_quarterly_2000_2024.csv`; then review `get_daily_price.py` and `market_data.py`.
3. Obtain the shock inputs from their original source. `Decomposition.py` expects `USMPD.csv` with `date_time`, `FF3`, and `SP500`. Its output is `jk_shocks_ff3_sp500.csv`; downstream filenames differ and must be aligned deliberately.
4. Review `impliedduration.py` and record its parameters. It currently writes `monthly_implied_duration_years.csv`, whereas `panel.py` expects `Implied_Duration_g=0.6.csv`. Do not infer the parameters of an old output from its filename alone.
5. `CAMP.py` expects `shocks_fed_jk_t.csv` and writes `forecasted_returns_5day_after_fomc_updated.csv`. `panel.py` instead expects `shocks.csv` and `abnormal_return.csv`. Reconcile names, schemas, and event-window definitions before proceeding.
6. Review and run `risk_factor.py`, then `panel.py` with consistent shock, duration, beta, and abnormal-return files. Outputs go to `panel_outputs_model_selectable`.
7. Point `increase_test.py` to the chosen coefficient output; their current default paths differ from `panel.py`'s output folder.

The saved aggregate tables in `results/` can be inspected without raw data. The full pipeline needs authorized inputs and configuration reconciliation; it has not been executed as part of packaging.
