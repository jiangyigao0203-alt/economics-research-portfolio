# File selection

This portfolio includes the main data-preparation and estimation scripts and a small set of aggregate results. Original files remain in the author's local workspace. Filenames and contents were inspected; a larger suffix or newer timestamp alone was not treated as proof that a script superseded another.

## 525

- Keep the monthly Compustat workflow, CRSP/market downloads, shock decomposition, `impliedduration.py`, diagnostics, event-return calculation, beta estimation, and configurable `panel.py`.
- Omit `getdatawrds.py`: annual-data branch outside the selected monthly workflow.
- Omit `binned_panel.py` and `TravisPlayground.py`: alternate panel implementations; `panel.py` exposes the selected model options.
- Omit `mothlyduration.py`: alternative duration implementation. The included `impliedduration.py` supports the optional NYSE microcap exclusion; the two implementations have different assumptions and are not interchangeable.
- Omit `Impliedduration_remove_negatives.py`: ad hoc filtering of a separate file outside the selected workflow.
- Omit `visualize.py` and helpers embedded in output folders: duplicate presentation utilities. Keep the root `increase_test.py` as the compact coefficient-pattern analysis.
- Keep only four aggregate tables from `panel_outputs_model_selectable`; omit firm-level panels, residuals, previews, logs, and redundant export formats.

## 575

- Keep `BSADF.py`, `start.py`, `arch2.py`, `model.py`, the three distinct MLE stages/experiments, and the ETF downloader.
- Omit `plot_all.py`: alternate lagged interval-union experiment; keep `start.py` as the selected zero-lag variant without implying equivalence across lag specifications.
- Omit `arch.py`: standalone GARCH plotting overlaps with `arch2.py` and estimation in the main models.
- Omit `Monte-Carlo.py` and `Plot.py`: separate calibration/simulation experiments with disconnected defaults; `start.py` includes its own calibration. This does not complete calibration for the standalone `BSADF.py` workflow.
- Omit `getSP.py` and `get_PE_PD.py`: valuation-data preparation branch outside the selected cross-market focus. `BSADF.py` retains a configurable valuation-series default, which must be supplied or changed.
- Omit `market_type.py`, `gsadf_summary.py`, and `visualize.py`: metadata editing and standalone presentation helpers outside the selected estimation workflow.
- Omit `prob.py`: unrelated coin-toss calculation.
- Replace the unlinked historical GSADF summary with one identified intensity-regression run's coefficients and model summary. Omit other historical runs, observation-level datasets, prediction dumps, and bulk charts.

These decisions curate an admissions code sample. They do not certify that each included file generated the archived outputs or that every excluded experiment is obsolete.
