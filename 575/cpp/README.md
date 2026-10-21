# C++ bubble-test simulations

Two standalone C++17 programs handle the computationally intensive tests and Monte Carlo calibration. Only source and build instructions are included; the archive's executables, raw datasets, generated output trees, and overlapping earlier implementations are excluded.

| Program | Purpose | Default configuration |
| --- | --- | --- |
| `multi_market_bsadf_gsadf.cpp` | Calibrate and calculate BSADF paths and GSADF tests for multiple markets | ADF lags 2 and 3; 500 null simulations; 95th-percentile critical values; five-observation minimum bubble duration |
| `bsadf_monte_carlo_lag0.cpp` | Dedicated zero-lag, endpoint-specific BSADF critical-value simulation | 6,289 observations; minimum window 383; 10,000 null simulations; 95th percentile |

The combined program comes from `ultimateADF.cpp`. The dedicated simulator comes from `monte_carlo.cpp`. Their statistical calculations are retained. The combined program's old absolute machine paths have been replaced with relative defaults, and comments explain the intended inputs and outputs.

## Build

Use a C++17 compiler and CMake. From this directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The combined program uses standard C++ threads. The dedicated simulator uses OpenMP when found by CMake and otherwise runs its simulation loop serially. There is no Eigen dependency. Executables are normally in `build/`; Visual Studio generators normally place Release executables in `build/Release/`.

Run with the process working directory set to `575/cpp`. For a single-configuration build on Linux or macOS:

```sh
./build/multi_market_bsadf_gsadf
./build/bsadf_monte_carlo_lag0
```

For a Visual Studio Release build in PowerShell:

```powershell
.\build\Release\multi_market_bsadf_gsadf.exe
.\build\Release\bsadf_monte_carlo_lag0.exe
```

Review the constants near the top of each source before running. The dedicated simulator's full defaults require about 174 billion window regressions and approximately 480 MiB for the simulation matrix alone. Smaller sample sizes and simulation counts are appropriate for a separate smoke-check configuration; they produce a different calibration and must not be confused with the full run.

These build instructions and source comparisons were reviewed, but compilation and execution were not verified in the packaging environment because a C++ compiler and CMake were not available on its command path.

## Combined-program inputs

The relative default root is `575/data/markets`, with these subdirectories:

```text
575/data/markets/
  ETFS/
  ExchangeRate/
  crypto/
```

Each CSV must have a header. Column 1 is a date string and column 2 is a strictly positive price; subsequent columns are ignored. The reader skips invalid or nonpositive observations and takes natural logarithms of retained prices. Files must already be chronological unless `SORT_BY_DATE_LEXICOGRAPHIC` is enabled with consistently sortable date strings. Daily and monthly observations are treated as ordered samples, so duration is measured in observations rather than a common calendar unit.

## Combined-program outputs

The relative default output parent is `575/outputs/bsadf_gsadf`. Each run creates a timestamped directory with:

- `BSADF_<market>_<type>_lag<lag>.csv`: endpoint statistics, critical values, raw flags, and duration-filtered `is_bubble` labels.
- `GSADF_summary_lag<lag>.csv`: one whole-sample test row per market for a given lag.

`target_metric` contains the original price; `price` contains its natural logarithm. This distinction matters because the Python feature and episode-regression scripts expect logged `price`. Early endpoints without an admissible test window have missing statistics. `is_bubble_raw` compares an endpoint statistic with its endpoint-specific critical value; `is_bubble` additionally removes runs shorter than the configured minimum. That duration filter is retrospective, so the labels should not be presented as real-time trading signals.

Calibration caches are reused only for matching sample size, minimum window, lag, simulation count, and quantile. BSADF calibration and GSADF calibration use different deterministic seed schedules. Endpoint critical values are not interchangeable with whole-sample critical values or the Python interval-union threshold.

## Connect to the Python analysis

Select one lag specification per analysis. Place the corresponding BSADF files into separate ETF, exchange-rate, and crypto input directories, then configure `INPUT_FOLDERS` in `../build_bubble_volatility_panel.py` and `../regress_bubble_episode_characteristics.py`. Do not accidentally mix lag-2 and lag-3 versions of the same series or feed GSADF summary tables into those scripts. Their required `date`, logged `price`, and `is_bubble` columns are present in the combined program's BSADF export.

This establishes the intended data interface. It does not establish that the newly supplied source version generated the repository's historical aggregate results.

## Dedicated zero-lag simulator

This program takes no external data. It simulates seeded Gaussian random walks and exports `part2_monte_carlo_cv_fast.csv` in the current working directory. The CSV contains `t_index` and `critical_value`. Minimum-window and sample-size settings must match any zero-lag test to which these critical values are applied. They do not calibrate lag-2 or lag-3 tests.
