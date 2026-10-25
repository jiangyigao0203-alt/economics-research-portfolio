# Data Availability

## Monetary-policy shocks and equity duration

We use Compustat accounting and security data, CRSP daily prices and linking information, and market/risk-free returns obtained through WRDS. Access requires the relevant institutional subscriptions. We do not redistribute the raw or firm-level WRDS extracts.

The monetary-policy and information-shock construction uses FOMC rate and equity surprises following the sign-restriction approach of Jarociński and Karadi (2020). The source files' required fields and the accounting-based duration parameters are specified in the project scripts. The optional NYSE size filter uses Kenneth French Data Library breakpoints.

## Financial bubbles across asset markets

The study uses cryptocurrency prices from CoinGecko, exchange-rate data from DataHub, and ETF prices from Yahoo Finance. The ETF universe follows ETF Database asset rankings. Coverage varies by series and extends to April 2026.

We provide per-series GSADF test results, market-category summaries, duration and intensity regressions, classifier metrics, the 50 random-market-split runs, and representative BSADF/GARCH figures. Raw price histories and large observation-level feature/prediction files are not included.

The C++ input schema is a dated positive-price CSV with date in column 1 and price in column 2. The Python analysis schema uses `date`, logged `price`, and `is_bubble`. Reproducing an analysis requires a consistent market universe, date coverage, lag specification, and calibration configuration.
