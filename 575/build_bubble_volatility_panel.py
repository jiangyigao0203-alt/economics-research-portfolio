"""Build an observation-level classification panel from bubble-test outputs.

Read calibrated BSADF result files with date, logged price, and is_bubble.
Estimate series-level GARCH volatility, construct lagged features and bubble
indicators, and export a panel plus data-quality summaries. This stage prepares
features; it does not fit the downstream logistic classifier. Review input
folders and frequency-dependent annualization before running.
"""

import json
import logging
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# SETTINGS
# =========================================================
INPUT_FOLDERS = {
    "ETF": Path(__file__).resolve().parent / 'ETFs_BSADF',
    "ExchangeRate": Path(__file__).resolve().parent / 'Exchange_Rate_BSADF',
    "crypto": Path(__file__).resolve().parent / 'cryptos_BSADF',
}

SEARCH_RECURSIVELY = True

BASE_OUTPUT_FOLDER = Path(__file__).resolve().parent / 'dataset_build_outputs'
BASE_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Required columns in each input CSV
DATE_COL = "date"
PRICE_COL = "price"         # already logged price
BUBBLE_COL = "is_bubble"

# Optional metadata columns
MARKET_NAME_COL_CANDIDATES = ["market_name", "series_name"]
MARKET_TYPE_COL_CANDIDATES = ["market_type"]

# =========================================================
# DATA / VOLATILITY CONFIG
# =========================================================
PRICE_IS_ALREADY_LOG = True

DAILY_ANNUALIZATION = 365
MONTHLY_ANNUALIZATION = 12
ANNUALIZE_VOL = True

MIN_TOTAL_OBS_FOR_FILE = 20
MIN_RET_OBS_FOR_GARCH = 20
MIN_ROWS_AFTER_FEATURES = 10

GARCH_PROGRESS_EVERY = 10

# =========================================================
# GLOBAL TIMER / LOGGER
# =========================================================
PROGRAM_START_TIME = time.time()


def now_str():
    return datetime.now(tz=UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def elapsed_str(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def log(msg):
    elapsed = time.time() - PROGRAM_START_TIME
    print(f"[{now_str()}][+{elapsed_str(elapsed)}] {msg}", flush=True)


def section(title):
    line = "=" * 100
    print(f"\n{line}", flush=True)
    print(title, flush=True)
    print(line, flush=True)


# =========================================================
# RUN FOLDER
# =========================================================
def create_run_output_folder(base_folder: Path):
    timestamp = datetime.now(tz=UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    run_folder = base_folder / f"run_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


# =========================================================
# GARCH(1,1) FUNCTIONS
# =========================================================
def compute_initial_variance(params, returns):
    """
    Use unconditional variance omega / (1 - alpha - beta) as the initial variance.
    Fall back to sample variance if invalid.
    """
    omega, alpha, beta = params
    persistence = alpha + beta

    if (
        not np.isfinite(omega)
        or not np.isfinite(alpha)
        or not np.isfinite(beta)
        or omega <= 0
        or persistence >= 0.999
    ):
        sample_var = np.var(returns)
        if not np.isfinite(sample_var) or sample_var <= 1e-12:
            sample_var = 1e-6
        return sample_var

    unconditional_var = omega / (1.0 - persistence)

    if not np.isfinite(unconditional_var) or unconditional_var <= 1e-12:
        sample_var = np.var(returns)
        if not np.isfinite(sample_var) or sample_var <= 1e-12:
            sample_var = 1e-6
        return sample_var

    return unconditional_var


def garch_filter(params, returns):
    omega, alpha, beta = params
    n = len(returns)
    variances = np.zeros(n)

    variances[0] = compute_initial_variance(params, returns)

    for t in range(1, n):
        variances[t] = omega + alpha * (returns[t - 1] ** 2) + beta * variances[t - 1]
        if variances[t] <= 1e-12 or not np.isfinite(variances[t]):
            variances[t] = 1e-12

    return variances


def garch_log_likelihood(params, returns):
    """Return the Gaussian GARCH negative log-likelihood used by numerical minimization."""
    omega, alpha, beta = params

    if omega <= 1e-8 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return 1e12

    variances = garch_filter(params, returns)

    if np.any(variances <= 0) or np.any(~np.isfinite(variances)):
        return 1e12

    log_lik = -0.5 * np.sum(
        np.log(2 * np.pi * variances) + (returns ** 2) / variances
    )

    if not np.isfinite(log_lik):
        return 1e12

    return -log_lik


class GarchProgressCallback:
    def __init__(self, returns, file_name, every=10):
        self.returns = returns
        self.file_name = file_name
        self.every = every
        self.iteration = 0
        self.start_time = time.time()

    def __call__(self, xk):
        self.iteration += 1
        if self.iteration % self.every != 0:
            return
        obj = garch_log_likelihood(xk, self.returns)
        elapsed = time.time() - self.start_time
        log(
            f"GARCH progress [{self.file_name}] iter={self.iteration} "
            f"negloglik={obj:.6f} params={np.round(xk, 6)} "
            f"elapsed={elapsed_str(elapsed)}"
        )


# =========================================================
# FREQUENCY DETECTION
# =========================================================
def detect_frequency(df, date_col):
    temp = df[[date_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce", utc=True)
    temp = temp.dropna().sort_values(date_col).reset_index(drop=True)

    if len(temp) < 5:
        return "unknown"

    diffs = temp[date_col].diff().dt.total_seconds() / (24 * 3600.0)
    diffs = diffs.dropna()

    if len(diffs) == 0:
        return "unknown"

    median_diff = np.median(diffs)

    if median_diff <= 2:
        return "daily"
    elif median_diff >= 20:
        return "monthly"
    else:
        mean_diff = np.mean(diffs)
        return "daily" if mean_diff <= 7 else "monthly"


# =========================================================
# BOOL / FILE HELPERS
# =========================================================
def normalize_bool_series(series):
    if series.dtype == bool:
        return series.fillna(False)

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    return mapped.fillna(False).astype(bool)


def get_first_existing_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def safe_market_name(df, file_path):
    col = get_first_existing_col(df, MARKET_NAME_COL_CANDIDATES)
    if col is not None and df[col].notna().any():
        val = str(df[col].dropna().iloc[0]).strip()
        if val:
            return val
    return file_path.stem


def list_csv_files(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return []

    if SEARCH_RECURSIVELY:
        return sorted(folder.rglob("*.csv"))
    return sorted(folder.glob("*.csv"))


# =========================================================
# PER-FILE DATASET CONSTRUCTION
# =========================================================
def process_file(file_path, configured_market_type):
    """Read one calibrated series, estimate GARCH variance, and construct aligned panel features and diagnostics."""
    file_start = time.time()
    log(f"Processing file start: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    log(f"Read CSV complete [{file_path.name}] rows={len(df)} cols={len(df.columns)}")

    required_cols = [DATE_COL, PRICE_COL, BUBBLE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log(f"Skipped [{file_path.name}] missing required columns: {missing}")
        return None, None, {
            "status": f"missing required columns: {missing}"
        }

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df[BUBBLE_COL] = normalize_bool_series(df[BUBBLE_COL])

    df = df.dropna(subset=[DATE_COL, PRICE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    log(f"Clean/sort complete [{file_path.name}] usable_rows={len(df)}")

    if len(df) < MIN_TOTAL_OBS_FOR_FILE:
        log(f"Skipped [{file_path.name}] too few valid observations")
        return df, None, {
            "status": "too few valid observations"
        }

    market_name = safe_market_name(df, file_path)
    market_type_col = get_first_existing_col(df, MARKET_TYPE_COL_CANDIDATES)

    if market_type_col is not None and df[market_type_col].notna().any():
        market_type = str(df[market_type_col].dropna().iloc[0]).strip()
    else:
        market_type = configured_market_type

    freq = detect_frequency(df, DATE_COL)
    log(f"Metadata [{file_path.name}] series_name={market_name} market_type={market_type} frequency={freq}")

    # Price is already logged
    df["log_price"] = df[PRICE_COL]

    # Returns from logged price
    df["log_return"] = df["log_price"].diff()

    df_ret = df.dropna(subset=["log_return"]).copy().reset_index(drop=False)

    if len(df_ret) < MIN_RET_OBS_FOR_GARCH:
        log(f"Skipped [{file_path.name}] too few return observations for GARCH")
        return df, None, {
            "status": "too few return observations for GARCH"
        }

    returns_raw = df_ret["log_return"].to_numpy(dtype=float)
    returns_scaled = returns_raw * 100.0

    v = np.var(returns_scaled)
    if not np.isfinite(v) or v <= 1e-12:
        v = 1e-6

    initial_params = [max(v * 0.05, 1e-6), 0.05, 0.90]
    log(
        f"GARCH fit start [{file_path.name}] n_returns={len(returns_scaled)} "
        f"initial_params={np.round(initial_params, 6)}"
    )

    garch_callback = GarchProgressCallback(
        returns=returns_scaled,
        file_name=file_path.name,
        every=GARCH_PROGRESS_EVERY
    )

    garch_start = time.time()
    opt_result = minimize(
        garch_log_likelihood,
        initial_params,
        args=(returns_scaled,),
        bounds=((1e-8, None), (0, 1), (0, 1)),
        method="SLSQP",
        callback=garch_callback,
        options={"maxiter": 200, "disp": False}
    )
    garch_elapsed = time.time() - garch_start

    if not opt_result.success:
        log(f"GARCH warning [{file_path.name}] optimizer message: {opt_result.message}")

    optimized_params = opt_result.x
    final_negloglik = garch_log_likelihood(optimized_params, returns_scaled)

    log(
        f"GARCH fit done [{file_path.name}] success={opt_result.success} "
        f"iters={getattr(opt_result, 'nit', 'NA')} "
        f"negloglik={final_negloglik:.6f} "
        f"params={np.round(optimized_params, 6)} "
        f"elapsed={elapsed_str(garch_elapsed)}"
    )

    cond_var_scaled = garch_filter(optimized_params, returns_scaled)
    df_ret["garch_vol"] = np.sqrt(cond_var_scaled) / 100.0

    if ANNUALIZE_VOL:
        annualization = MONTHLY_ANNUALIZATION if freq == "monthly" else DAILY_ANNUALIZATION
        df_ret["garch_vol"] = df_ret["garch_vol"] * np.sqrt(annualization)
        log(f"Annualized volatility [{file_path.name}] factor=sqrt({annualization})")

    df = df.merge(df_ret[["index", "garch_vol"]], left_index=True, right_on="index", how="left")
    df = df.drop(columns=["index"])
    log(f"Merged GARCH vol back [{file_path.name}]")

    # Bubble start definition: first True after False
    df["bubble_lag1"] = df[BUBBLE_COL].shift(1).fillna(False).astype(bool)
    df["bubble_start"] = ((df[BUBBLE_COL] == True) & (df["bubble_lag1"] == False)).astype(int)

    # Lagged volatility predictor
    df["garch_vol_lag1"] = df["garch_vol"].shift(1)

    # Metadata
    df["series_name"] = market_name
    df["market_type"] = market_type
    df["frequency"] = freq
    df["source_file"] = file_path.name
    df["source_path"] = str(file_path)
    df["year"] = df[DATE_COL].dt.year.astype("Int64")

    # Keep rows where lagged volatility exists
    usable = df.dropna(subset=["garch_vol_lag1"]).copy()

    if len(usable) < MIN_ROWS_AFTER_FEATURES:
        log(f"Skipped [{file_path.name}] too few usable rows after lagged features")
        return df, None, {
            "status": "too few usable rows after lagged features"
        }

    metadata = {
        "status": "ok",
        "series_name": market_name,
        "market_type": market_type,
        "frequency": freq,
        "n_full_rows": len(df),
        "n_usable_rows": len(usable),
        "n_bubble_starts": int(usable["bubble_start"].sum()),
        "garch_success": bool(opt_result.success),
        "garch_iterations": int(getattr(opt_result, "nit", -1)) if getattr(opt_result, "nit", None) is not None else None,
        "garch_omega": float(optimized_params[0]),
        "garch_alpha": float(optimized_params[1]),
        "garch_beta": float(optimized_params[2]),
        "garch_persistence": float(optimized_params[1] + optimized_params[2]),
        "garch_negloglik": float(final_negloglik),
        "garch_initial_variance": float(compute_initial_variance(optimized_params, returns_scaled)),
    }

    log(
        f"File complete [{file_path.name}] usable_rows={len(usable)} "
        f"bubble_starts={int(usable['bubble_start'].sum())} "
        f"elapsed={elapsed_str(time.time() - file_start)}"
    )

    return df, usable, metadata


# =========================================================
# BUILD POOLED PANEL DATASET
# =========================================================
def build_panel_dataset(input_folders):
    """Combine usable per-series observations into the classifier feature panel."""
    all_rows = []
    file_level_status_rows = []

    for configured_market_type, folder in input_folders.items():
        folder = Path(folder)

        section(f"SCANNING FOLDER: {folder} | CONFIGURED MARKET TYPE: {configured_market_type}")
        log(f"Folder exists={folder.exists()} is_dir={folder.is_dir()}")

        csv_files = list_csv_files(folder)
        log(f"CSV files found in folder: {len(csv_files)}")

        if csv_files:
            preview_n = min(10, len(csv_files))
            log(f"Previewing first {preview_n} file(s)")
            for f in csv_files[:preview_n]:
                log(f"  {f}")

        if not csv_files:
            log(f"No CSV files found in {folder}")
            continue

        for file_idx, file_path in enumerate(csv_files, start=1):
            log(f"Folder progress [{configured_market_type}] file {file_idx}/{len(csv_files)} -> {file_path.name}")

            try:
                full_df, usable_df, meta = process_file(file_path, configured_market_type)

                file_level_status_rows.append({
                    "configured_market_type": configured_market_type,
                    "folder": str(folder),
                    "file": file_path.name,
                    "file_path": str(file_path),
                    "full_rows": 0 if full_df is None else len(full_df),
                    "usable_rows": 0 if usable_df is None else len(usable_df),
                    "bubble_start_rows": 0 if usable_df is None else int(usable_df["bubble_start"].sum()),
                    **meta,
                })

                if usable_df is not None and not usable_df.empty:
                    all_rows.append(usable_df)

            except Exception as e:
                logging.getLogger(__name__).exception("Operation failed in %s", __file__)
                log(f"ERROR processing {file_path.name}: {e}")
                file_level_status_rows.append({
                    "configured_market_type": configured_market_type,
                    "folder": str(folder),
                    "file": file_path.name,
                    "file_path": str(file_path),
                    "full_rows": 0,
                    "usable_rows": 0,
                    "bubble_start_rows": 0,
                    "status": f"error: {e}",
                })

    status_df = pd.DataFrame(file_level_status_rows)

    if not all_rows:
        raise ValueError("No usable panel observations were created from the configured folders.")

    panel_df = pd.concat(all_rows, ignore_index=True)
    log(f"Panel dataset built. Total rows={len(panel_df)}")
    return panel_df, status_df


# =========================================================
# PANEL CLEANUP / FINAL COLUMN SELECTION
# =========================================================
def finalize_panel_dataset(panel_df):
    """Select the exported feature columns and sort observations by series and date."""
    keep_cols = [
        DATE_COL,
        "series_name",
        "market_type",
        "frequency",
        "source_file",
        "source_path",
        PRICE_COL,
        "log_price",
        "log_return",
        BUBBLE_COL,
        "bubble_lag1",
        "bubble_start",
        "garch_vol",
        "garch_vol_lag1",
        "year",
    ]

    keep_cols = [c for c in keep_cols if c in panel_df.columns]
    out = panel_df[keep_cols].copy()

    out = out.sort_values(["series_name", DATE_COL]).reset_index(drop=True)

    return out


# =========================================================
# SUMMARIES
# =========================================================
def build_dataset_summaries(panel_df):
    overall_summary = {
        "n_rows": len(panel_df),
        "n_series": int(panel_df["series_name"].nunique()) if "series_name" in panel_df.columns else None,
        "n_market_types": int(panel_df["market_type"].nunique()) if "market_type" in panel_df.columns else None,
        "n_bubble_starts": int(panel_df["bubble_start"].sum()) if "bubble_start" in panel_df.columns else None,
        "bubble_start_rate": float(panel_df["bubble_start"].mean()) if "bubble_start" in panel_df.columns else None,
    }

    by_market_type = (
        panel_df.groupby("market_type", dropna=False)
        .agg(
            n_rows=("bubble_start", "size"),
            n_series=("series_name", "nunique"),
            n_bubble_starts=("bubble_start", "sum"),
            bubble_start_rate=("bubble_start", "mean"),
            mean_garch_vol=("garch_vol", "mean"),
            mean_garch_vol_lag1=("garch_vol_lag1", "mean"),
        )
        .reset_index()
    )

    by_series = (
        panel_df.groupby(["series_name", "market_type"], dropna=False)
        .agg(
            n_rows=("bubble_start", "size"),
            first_date=(DATE_COL, "min"),
            last_date=(DATE_COL, "max"),
            n_bubble_starts=("bubble_start", "sum"),
            bubble_start_rate=("bubble_start", "mean"),
            mean_garch_vol=("garch_vol", "mean"),
            mean_garch_vol_lag1=("garch_vol_lag1", "mean"),
        )
        .reset_index()
    )

    by_frequency = (
        panel_df.groupby("frequency", dropna=False)
        .agg(
            n_rows=("bubble_start", "size"),
            n_series=("series_name", "nunique"),
            n_bubble_starts=("bubble_start", "sum"),
            bubble_start_rate=("bubble_start", "mean"),
        )
        .reset_index()
    )

    return overall_summary, by_market_type, by_series, by_frequency


# =========================================================
# SAVE OUTPUTS
# =========================================================
def save_outputs(
    output_folder,
    panel_df,
    status_df,
    overall_summary,
    by_market_type,
    by_series,
    by_frequency,
):
    panel_file = output_folder / "panel_dataset_full.csv"
    status_file = output_folder / "file_processing_status.csv"
    overall_summary_file = output_folder / "overall_summary.json"
    by_market_type_file = output_folder / "summary_by_market_type.csv"
    by_series_file = output_folder / "summary_by_series.csv"
    by_frequency_file = output_folder / "summary_by_frequency.csv"
    config_file = output_folder / "dataset_build_config.txt"
    run_info_file = output_folder / "run_info.txt"

    log("Saving panel dataset")
    panel_df.to_csv(panel_file, index=False)

    log("Saving file processing status")
    status_df.to_csv(status_file, index=False)

    log("Saving overall summary JSON")
    with open(overall_summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, default=str)

    log("Saving summaries by market type / series / frequency")
    by_market_type.to_csv(by_market_type_file, index=False)
    by_series.to_csv(by_series_file, index=False)
    by_frequency.to_csv(by_frequency_file, index=False)

    log("Saving config")
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("DATASET BUILD CONFIG\n")
        f.write("=" * 80 + "\n")
        f.write(f"INPUT_FOLDERS = {INPUT_FOLDERS}\n")
        f.write(f"SEARCH_RECURSIVELY = {SEARCH_RECURSIVELY}\n")
        f.write(f"DATE_COL = {DATE_COL}\n")
        f.write(f"PRICE_COL = {PRICE_COL}\n")
        f.write(f"BUBBLE_COL = {BUBBLE_COL}\n")
        f.write(f"PRICE_IS_ALREADY_LOG = {PRICE_IS_ALREADY_LOG}\n")
        f.write(f"ANNUALIZE_VOL = {ANNUALIZE_VOL}\n")
        f.write(f"DAILY_ANNUALIZATION = {DAILY_ANNUALIZATION}\n")
        f.write(f"MONTHLY_ANNUALIZATION = {MONTHLY_ANNUALIZATION}\n")
        f.write(f"MIN_TOTAL_OBS_FOR_FILE = {MIN_TOTAL_OBS_FOR_FILE}\n")
        f.write(f"MIN_RET_OBS_FOR_GARCH = {MIN_RET_OBS_FOR_GARCH}\n")
        f.write(f"MIN_ROWS_AFTER_FEATURES = {MIN_ROWS_AFTER_FEATURES}\n")
        f.write(f"GARCH_PROGRESS_EVERY = {GARCH_PROGRESS_EVERY}\n")
        f.write("GARCH initial variance = unconditional variance omega / (1 - alpha - beta), with fallback to sample variance if invalid.\n")
        f.write(f"Rows in final panel = {len(panel_df)}\n")
        f.write(f"Bubble starts in final panel = {int(panel_df['bubble_start'].sum())}\n")

    log("Saving run info")
    with open(run_info_file, "w", encoding="utf-8") as f:
        f.write(f"Run timestamp: {datetime.now(tz=UTC).astimezone()}\n")
        f.write(f"Output folder: {output_folder}\n")
        f.write(f"Rows in final panel: {len(panel_df)}\n")
        f.write(f"Bubble starts: {int(panel_df['bubble_start'].sum())}\n")

    print("\nSaved files:", flush=True)
    print(f"  Panel dataset           : {panel_file}", flush=True)
    print(f"  File processing status  : {status_file}", flush=True)
    print(f"  Overall summary         : {overall_summary_file}", flush=True)
    print(f"  Summary by market type  : {by_market_type_file}", flush=True)
    print(f"  Summary by series       : {by_series_file}", flush=True)
    print(f"  Summary by frequency    : {by_frequency_file}", flush=True)
    print(f"  Config                  : {config_file}", flush=True)
    print(f"  Run info                : {run_info_file}", flush=True)


# =========================================================
# PRINT RESULTS
# =========================================================
def print_dataset_preview(panel_df, overall_summary, by_market_type, by_series, by_frequency):
    section("FINAL PANEL DATASET CREATED")
    log(f"Rows in panel: {overall_summary['n_rows']}")
    log(f"Series count: {overall_summary['n_series']}")
    log(f"Market type count: {overall_summary['n_market_types']}")
    log(f"Bubble starts: {overall_summary['n_bubble_starts']}")
    log(f"Bubble start rate: {overall_summary['bubble_start_rate']:.8f}")

    section("PANEL DATASET HEAD")
    print(panel_df.head(20).to_string(index=False), flush=True)

    section("SUMMARY BY MARKET TYPE")
    print(by_market_type.to_string(index=False), flush=True)

    section("SUMMARY BY FREQUENCY")
    print(by_frequency.to_string(index=False), flush=True)

    section("TOP 20 SERIES BY BUBBLE START COUNT")
    if not by_series.empty:
        top_series = by_series.sort_values("n_bubble_starts", ascending=False).head(20)
        print(top_series.to_string(index=False), flush=True)


# =========================================================
# MAIN
# =========================================================
def main():
    section("PROGRAM START")
    run_output_folder = create_run_output_folder(BASE_OUTPUT_FOLDER)
    log(f"Output folder for this run: {run_output_folder}")

    log("Building pooled panel dataset")
    panel_df_raw, status_df = build_panel_dataset(INPUT_FOLDERS)

    log("Finalizing panel dataset columns/order")
    panel_df = finalize_panel_dataset(panel_df_raw)

    log("Building summary tables")
    overall_summary, by_market_type, by_series, by_frequency = build_dataset_summaries(panel_df)

    print_dataset_preview(panel_df, overall_summary, by_market_type, by_series, by_frequency)

    save_outputs(
        output_folder=run_output_folder,
        panel_df=panel_df,
        status_df=status_df,
        overall_summary=overall_summary,
        by_market_type=by_market_type,
        by_series=by_series,
        by_frequency=by_frequency,
    )

    section("PROGRAM COMPLETE")
    log("Dataset construction finished successfully")


if __name__ == "__main__":
    main()
