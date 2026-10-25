# Research in Financial Economics

I maintain code and empirical results for two coauthored studies of monetary-policy transmission and financial bubbles.

| Research project | Authors | Contents |
| --- | --- | --- |
| [High-Frequency Identification of Monetary Policy and Information Shocks Across Implied Equity Durations](equity-duration-and-monetary-policy-shocks/) | Yigao Jiang and Travis J. Kimball | Shock identification, equity-duration estimation, event returns, and panel regressions |
| [Identifying Financial Bubbles Across Heterogeneous Asset Markets: Evidence from Cryptocurrencies, ETFs, and Exchange Rates](financial-bubbles-across-asset-markets/) | Yigao Jiang and Matthew Lam | GSADF/BSADF tests, Monte Carlo calibration, GARCH volatility, duration and intensity regressions, and logit prediction experiments |

Each project includes a method guide and empirical result tables. The bubble study also includes representative identification figures, chronological-holdout metrics, and all 50 random-market-holdout runs.

## Python environment

The Python scripts require Python 3.11 or later. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run analysis scripts from their project directory after configuring their input paths. C++ build instructions are in the [bubble-test guide](financial-bubbles-across-asset-markets/cpp/).

## Data

The repository distributes code, aggregate results, and selected figures. Raw WRDS/Compustat/CRSP data require authorized access and are not redistributed. External price and shock sources are described in [Data availability](DATA_AVAILABILITY.md).
