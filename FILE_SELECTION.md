# File selection

The portfolio includes research source, documentation, and a small set of aggregate results. Original workspace files and the supplied archive are preserved. Source names are mapped in [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md).

## 525

Keep the monthly Compustat workflow, CRSP/market downloads, shock decomposition, implied-duration estimation, diagnostics, event-return calculation, beta estimation, configurable panel model, and coefficient trend check. Exclude the annual-data branch (`getdatawrds.py`), alternate panel implementations (`binned_panel.py`, `TravisPlayground.py`), alternative duration implementation (`mothlyduration.py`), ad hoc negative-duration filtering, and duplicate presentation helpers.

Keep four aggregate tables from the selected panel-output folder. Exclude firm-level panels, residuals, previews, logs, and redundant export formats. Different historical duration implementations have different assumptions; exclusion does not establish numerical equivalence.

## 575: Python

Keep the standalone BSADF calculation, zero-lag interval-union experiment, GARCH visualization, episode model, feature builder, two distinct classifier experiments, and ETF downloader. Exclude the alternate lagged union experiment (`plot_all.py`), earlier GARCH plot (`arch.py`), disconnected standalone calibration/simulation experiments (`Monte-Carlo.py`, `Plot.py`), valuation-data download branch, metadata/presentation helpers, and unrelated coin-toss helper (`prob.py`).

Retain one identified intensity-regression run's coefficients and model summary. Exclude bulk charts, prediction dumps, observation-level datasets, and the previously unlinked GSADF summary.

## 575: C++ archive

Keep `ultimateADF.cpp` as `cpp/multi_market_bsadf_gsadf.cpp`: the combined multi-market BSADF/GSADF workflow supports configurable lag lists and includes its own calibration. Keep `monte_carlo.cpp` as `cpp/bsadf_monte_carlo_lag0.cpp`: a distinct zero-lag simulator configured for a larger 10,000-path run.

Exclude the overlapping single-purpose programs `bsadf.cpp`, `multi_market_bsadf.cpp`, and `multi_market_gsadf.cpp`; exclude `main.cpp`, which is a counting benchmark. Exclude every `.exe`, editor-specific settings, raw-data directory, and generated result tree. The published CMake build uses standard threads and optional OpenMP rather than machine-specific compiler paths or unused Eigen flags.

Selection is based on purpose and overlapping functionality, not a claim that every excluded experiment is obsolete. New C++ sources have not been established as the source of the archived Python regression outputs.
