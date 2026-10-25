# High-Frequency Identification of Monetary Policy and Information Shocks Across Implied Equity Durations

**Yigao Jiang and Travis J. Kimball — April 2026**

We study how equity prices respond to pure monetary-policy surprises and central-bank information shocks across firms with different implied cash-flow durations. We combine high-frequency FOMC shock identification with accounting-based equity duration and post-announcement cumulative abnormal returns.

Our results show a stronger duration gradient for the information shock than for the pure monetary-policy shock. The available panel contains 965,551 firm-event observations, 13,890 firms, and 193 events.

## Methods and source files

| Stage | Source |
| --- | --- |
| Monthly Compustat accounting/security panel | `build_compustat_monthly_panel.py` |
| Matched CRSP prices | `download_crsp_daily_prices.py` |
| Market and risk-free returns | `download_market_and_risk_free_returns.py` |
| Sign-restricted policy/information shock decomposition | `decompose_monetary_policy_shocks.py` |
| Implied cash-flow duration and diagnostics | `estimate_equity_duration.py`, `diagnose_equity_duration.py` |
| Event-window abnormal returns | `estimate_event_abnormal_returns.py` |
| Firm market betas | `estimate_asset_market_betas.py` |
| Duration interactions and fixed-effect specifications | `estimate_duration_shock_panel.py` |
| Trends across duration-bin coefficients | `test_duration_coefficient_trend.py` |

## Results

- [Panel coefficients](results/panel_regression_coefficients.csv)
- [Model statistics](results/panel_regression_model_stats.csv)
- [Sample statistics](results/panel_regression_sample_stats.csv)
- [Duration-bin coefficient trends](results/beta_monotonicity_test_results.csv)

The panel specification includes firm fixed effects, duration-quantile interactions, asset-beta controls, and HC1 covariance. The coefficient-trend calculation uses inverse marginal-variance weights; it does not propagate the full first-stage coefficient covariance.

## Running the analysis

The scripts' configuration sections define data sources, dates, parameters, model choices, and output filenames. The monthly accounting panel feeds both price matching and duration estimation. The event-return calculation and market-beta calculation use matched daily prices and market returns. The final panel merges event returns, shock dates, lagged duration, and betas.

Input filenames identify separate analysis configurations. For example, the duration estimator writes `monthly_implied_duration_years.csv`, while the panel's configured input is `Implied_Duration_g=0.6.csv`. The shock decomposition writes `jk_shocks_ff3_sp500.csv`; the event and panel scripts have independently configured shock filenames. Set these paths consistently for the intended run and record the duration assumptions used.

The FOMC shocks use a narrow intraday identification window, whereas the equity CARs use daily data over a post-event window. Concurrent macroeconomic news can therefore affect the return estimates. WRDS collection requires Compustat and CRSP access; raw firm-level data are not included.
