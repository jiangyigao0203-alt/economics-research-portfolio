# Bubble Tests and Monte Carlo Calibration

We use C++17 implementations for recursive ADF testing and simulation-based critical values.

| Program | Role | Default settings |
| --- | --- | --- |
| `multi_market_gsadf.cpp` | Standalone whole-series GSADF test for one input directory | One lagged difference; 500 simulations; 95th percentile |
| `multi_market_bsadf_gsadf.cpp` | Combined global GSADF and dated BSADF analysis across market categories | Lag list 2 and 3; 500 simulations; five-observation minimum episode |
| `bsadf_monte_carlo_lag0.cpp` | Dedicated zero-lag BSADF critical-value path | 6,289 observations; minimum window 383; 10,000 simulations |

The standalone GSADF implementation uses prefix sums specialized to one lag. Changing its `ADF_LAGS` label does not implement another lag order. The combined program supports an explicit lag list. Critical values must match the sample length, minimum window, and lag specification of the associated test.

## Build

From this directory, with a C++17 compiler and CMake:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The GSADF programs use standard C++ threads. The dedicated zero-lag simulator uses OpenMP when available and otherwise runs its simulation loop serially. No Eigen dependency is required.

For a single-configuration build, run `./build/multi_market_gsadf` or `./build/multi_market_bsadf_gsadf` from this directory. For a Visual Studio Release build in PowerShell, use `.\build\Release\multi_market_gsadf.exe` or `.\build\Release\multi_market_bsadf_gsadf.exe`.

## Price inputs

The combined program reads `../data/markets/ETFS`, `../data/markets/ExchangeRate`, and `../data/markets/crypto`. The standalone GSADF default reads only `../data/markets/ETFS`; set `DATASET_DIR` for the other categories.

Each CSV must contain a header, date in column 1, and positive price in column 2. Subsequent columns are ignored. Prices are logged internally; invalid/nonpositive rows are skipped. Observations must be chronological unless string-date sorting is enabled with consistently sortable dates.

## Outputs

The standalone program writes a timestamped `market_summary.csv` under `../outputs/gsadf`. It reports the statistic, simulated critical value, rejection, sample length, and date range for each series. The combined program writes BSADF paths and `GSADF_summary_lag<lag>.csv` files under `../outputs/bsadf_gsadf`.

In combined BSADF files, `target_metric` is the raw price and `price` is its natural logarithm. `is_bubble_raw` compares the endpoint statistic with its critical value. `is_bubble` additionally removes runs shorter than the minimum duration, using the eventual length of each run. These filtered labels are retrospective.

The Python episode and feature scripts consume `date`, logged `price`, and `is_bubble`. Select one lag specification and separate the market categories in their configured input folders. GSADF summary rows are not episode-level inputs.

## Dedicated calibration

`bsadf_monte_carlo_lag0` needs no external data and writes `part2_monte_carlo_cv_fast.csv` in the working directory. Its defaults imply roughly 174 billion regression windows and about 480 MiB for the simulation matrix alone. Sample size, minimum window, simulation count, and seeds are defined near the top of the source.
