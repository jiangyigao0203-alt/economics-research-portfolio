"""Regress bubble-episode duration or intensity on volatility and controls.

Read calibrated BSADF result CSVs with date, logged price, and is_bubble.
Estimate GARCH volatility by series, identify contiguous bubble episodes, and
build episode-level outcomes and regressors. DEPENDENT_VARIABLE selects the
outcome; optional market-type/year controls are configured below. OLS uses HC1
covariance. Save estimates and diagnostics in a timestamped run directory.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

# =========================================================
# SETTINGS
# =========================================================

# Three configurable folders. Each folder should contain BSADF result CSV files.
# SEARCH_RECURSIVELY includes result files in nested input directories.
INPUT_FOLDERS = {
    "ETF": Path(__file__).resolve().parent / 'ETFs_BSADF',
    "ExchangeRate": Path(__file__).resolve().parent / 'Exchange_Rate_BSADF',
    "crypto": Path(__file__).resolve().parent / 'cryptos_BSADF',
}

SEARCH_RECURSIVELY = True

BASE_OUTPUT_FOLDER = Path(__file__).resolve().parent / 'final_regression_outputs'
BASE_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Column names expected in the BSADF output files
DATE_COL = "date"
PRICE_COL = "price"         # already logged price
BUBBLE_COL = "is_bubble"

# Optional fallback columns if needed
MARKET_NAME_COL_CANDIDATES = ["market_name", "series_name"]
MARKET_TYPE_COL_CANDIDATES = ["market_type"]

# =========================================================
# DEPENDENT VARIABLE CONFIG
# =========================================================
# Choose:
#   "duration"  -> bubble duration in observations
#   "intensity" -> max(log_price) - min(log_price) during the bubble
DEPENDENT_VARIABLE = "intensity"

# =========================================================
# VOLATILITY MODEL CONFIG
# =========================================================
# Price is already logged, so returns are diff(price)
PRICE_IS_ALREADY_LOG = True

# Annualization
DAILY_ANNUALIZATION = 365
MONTHLY_ANNUALIZATION = 12
ANNUALIZE_VOL = True

# Choose which regressors to include
INCLUDE_PRE_BUBBLE_AVG_VOL = False
INCLUDE_VOL_RIGHT_BEFORE_BUBBLE = True
INCLUDE_MARKET_AVG_VOL = False
INCLUDE_MARKET_TYPE_DUMMIES = True
INCLUDE_YEAR_DUMMIES = False

# Pre-bubble volatility window settings
# "fixed"  -> use PRE_WINDOW_FIXED observations
# "scaled" -> use PRE_WINDOW_N_MULTIPLIER * bubble_duration observations
PRE_WINDOW_MODE = "scaled"   # "fixed" or "scaled"
PRE_WINDOW_FIXED = 20
PRE_WINDOW_N_MULTIPLIER = 1.0

# =========================================================
# DATA FILTERS
# =========================================================
MIN_BUBBLE_DURATION = 5
MIN_TOTAL_OBS_FOR_FILE = 20
MIN_RET_OBS_FOR_GARCH = 20

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
    Fall back to sample variance if persistence is too close to 1 or invalid.
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
# BUBBLE IDENTIFICATION
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


def identify_bubble_periods(df, bubble_col):
    """Identify contiguous runs of true bubble indicators in chronological order."""
    bubble = normalize_bool_series(df[bubble_col])

    if bubble.empty or not bubble.any():
        return []

    starts = bubble & ~bubble.shift(1, fill_value=False)
    ends = bubble & ~bubble.shift(-1, fill_value=False)

    start_idx = list(df.index[starts])
    end_idx = list(df.index[ends])

    periods = []
    for s, e in zip(start_idx, end_idx):
        duration_obs = e - s + 1
        periods.append({
            "start_idx": s,
            "end_idx": e,
            "duration_obs": duration_obs,
        })
    return periods


# =========================================================
# FILE HELPERS
# =========================================================
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
# PER-FILE PROCESSING
# =========================================================
def process_file(file_path, configured_market_type):
    """Read one calibrated series, estimate GARCH variance, and summarize its contiguous bubble episodes."""
    print(f"\nProcessing file: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    required_cols = [DATE_COL, PRICE_COL, BUBBLE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  Skipped: missing required columns {missing}")
        return None, None

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df[BUBBLE_COL] = normalize_bool_series(df[BUBBLE_COL])

    df = df.dropna(subset=[DATE_COL, PRICE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    if len(df) < MIN_TOTAL_OBS_FOR_FILE:
        print("  Skipped: too few valid observations")
        return None, None

    market_name = safe_market_name(df, file_path)
    market_type_col = get_first_existing_col(df, MARKET_TYPE_COL_CANDIDATES)

    if market_type_col is not None and df[market_type_col].notna().any():
        market_type = str(df[market_type_col].dropna().iloc[0]).strip()
    else:
        market_type = configured_market_type

    freq = detect_frequency(df, DATE_COL)

    # price already logged
    df["log_price"] = df[PRICE_COL]

    # returns from logged price
    df["log_return"] = df["log_price"].diff()

    df_ret = df.dropna(subset=["log_return"]).copy().reset_index(drop=False)
    # original row index preserved in 'index'

    if len(df_ret) < MIN_RET_OBS_FOR_GARCH:
        print("  Skipped: too few return observations for GARCH")
        return df, None

    returns_raw = df_ret["log_return"].to_numpy(dtype=float)
    returns_scaled = returns_raw * 100.0

    v = np.var(returns_scaled)
    if not np.isfinite(v) or v <= 1e-12:
        v = 1e-6

    initial_params = [max(v * 0.05, 1e-6), 0.05, 0.90]

    opt_result = minimize(
        garch_log_likelihood,
        initial_params,
        args=(returns_scaled,),
        bounds=((1e-8, None), (0, 1), (0, 1)),
        method="SLSQP"
    )

    if not opt_result.success:
        print(f"  Warning: GARCH optimizer message: {opt_result.message}")

    optimized_params = opt_result.x
    cond_var_scaled = garch_filter(optimized_params, returns_scaled)
    df_ret["garch_vol"] = np.sqrt(cond_var_scaled) / 100.0

    if ANNUALIZE_VOL:
        annualization = MONTHLY_ANNUALIZATION if freq == "monthly" else DAILY_ANNUALIZATION
        df_ret["garch_vol"] = df_ret["garch_vol"] * np.sqrt(annualization)

    # Merge volatility back to main dataframe
    df = df.merge(df_ret[["index", "garch_vol"]], left_index=True, right_on="index", how="left")
    df = df.drop(columns=["index"])

    market_avg_vol = df["garch_vol"].mean(skipna=True)

    periods = identify_bubble_periods(df, BUBBLE_COL)
    if not periods:
        print("  No bubbles found")
        return df, None

    bubble_rows = []
    for bubble_id, p in enumerate(periods, start=1):
        start_idx = p["start_idx"]
        end_idx = p["end_idx"]
        duration_obs = p["duration_obs"]

        if duration_obs < MIN_BUBBLE_DURATION:
            continue

        bubble_slice = df.loc[start_idx:end_idx].copy()

        # Year dummy basis: bubble start year
        bubble_start_date = df.loc[start_idx, DATE_COL]
        bubble_end_date = df.loc[end_idx, DATE_COL]
        if not isinstance(bubble_start_date, pd.Timestamp):
            raise TypeError("Bubble start date must be a parsed timestamp")
        bubble_year = bubble_start_date.year

        # Intensity = max(log_price) - min(log_price) during bubble
        bubble_intensity = bubble_slice["log_price"].max() - bubble_slice["log_price"].min()

        # Volatility right before bubble
        vol_right_before = np.nan
        if start_idx - 1 >= 0:
            vol_right_before = df.loc[start_idx - 1, "garch_vol"]

        # Pre-bubble average vol window
        if PRE_WINDOW_MODE == "fixed":
            pre_window_len = int(PRE_WINDOW_FIXED)
        elif PRE_WINDOW_MODE == "scaled":
            pre_window_len = max(1, round(PRE_WINDOW_N_MULTIPLIER * duration_obs))
        else:
            raise ValueError("PRE_WINDOW_MODE must be 'fixed' or 'scaled'")

        pre_start = max(0, start_idx - pre_window_len)
        pre_end = start_idx - 1

        if pre_end >= pre_start:
            pre_bubble_avg_vol = df.loc[pre_start:pre_end, "garch_vol"].mean(skipna=True)
        else:
            pre_bubble_avg_vol = np.nan

        bubble_rows.append({
            "source_file": file_path.name,
            "source_path": str(file_path),
            "series_name": market_name,
            "market_type": market_type,
            "frequency": freq,

            "bubble_id": bubble_id,
            "bubble_start_date": bubble_start_date,
            "bubble_end_date": bubble_end_date,
            "bubble_start_year": bubble_year,

            "bubble_start_idx": start_idx,
            "bubble_end_idx": end_idx,

            "bubble_duration_obs": duration_obs,
            "bubble_intensity": bubble_intensity,

            "pre_window_mode": PRE_WINDOW_MODE,
            "pre_window_length_obs": pre_window_len,

            "pre_bubble_avg_vol": pre_bubble_avg_vol,
            "vol_right_before_bubble": vol_right_before,
            "market_avg_vol": market_avg_vol,

            "garch_omega": optimized_params[0],
            "garch_alpha": optimized_params[1],
            "garch_beta": optimized_params[2],
            "garch_persistence": optimized_params[1] + optimized_params[2],
            "garch_initial_variance": compute_initial_variance(optimized_params, returns_scaled),
        })

    bubble_df = pd.DataFrame(bubble_rows)
    print(f"  Bubble rows created: {len(bubble_df)}")

    return df, bubble_df


# =========================================================
# BUILD POOLED REGRESSION DATASET
# =========================================================
def build_regression_dataset(input_folders):
    """Combine per-series episode records and file-processing diagnostics."""
    all_bubbles = []
    file_level_status_rows = []

    for configured_market_type, folder in input_folders.items():
        folder = Path(folder)
        print("\n" + "=" * 80)
        print(f"SCANNING FOLDER: {folder}")
        print(f"CONFIGURED MARKET TYPE: {configured_market_type}")
        print("=" * 80)

        print(f"  Exists: {folder.exists()}")
        print(f"  Is dir: {folder.is_dir()}")

        csv_files = list_csv_files(folder)
        print(f"  CSV files found: {len(csv_files)}")

        if csv_files:
            preview_n = min(10, len(csv_files))
            print(f"  First {preview_n} file(s):")
            for f in csv_files[:preview_n]:
                print(f"    {f}")

        if not csv_files:
            print(f"  No CSV files found in {folder}")
            continue

        for file_path in csv_files:
            try:
                df, bubble_df = process_file(file_path, configured_market_type)

                file_level_status_rows.append({
                    "configured_market_type": configured_market_type,
                    "folder": str(folder),
                    "file": file_path.name,
                    "file_path": str(file_path),
                    "usable_rows": 0 if df is None else len(df),
                    "bubble_rows": 0 if bubble_df is None else len(bubble_df),
                    "status": "ok" if bubble_df is not None else "no_bubbles_or_skipped",
                })

                if bubble_df is not None and not bubble_df.empty:
                    all_bubbles.append(bubble_df)

            except Exception as e:
                logging.getLogger(__name__).exception("Operation failed in %s", __file__)
                print(f"  ERROR processing {file_path.name}: {e}")
                file_level_status_rows.append({
                    "configured_market_type": configured_market_type,
                    "folder": str(folder),
                    "file": file_path.name,
                    "file_path": str(file_path),
                    "usable_rows": 0,
                    "bubble_rows": 0,
                    "status": f"error: {e}",
                })

    status_df = pd.DataFrame(file_level_status_rows)

    if not all_bubbles:
        raise ValueError("No bubble-level observations were created from the configured folders.")

    reg_df = pd.concat(all_bubbles, ignore_index=True)
    return reg_df, status_df


# =========================================================
# DESIGN MATRIX
# =========================================================
def get_dependent_variable_column():
    if DEPENDENT_VARIABLE == "duration":
        return "bubble_duration_obs"
    if DEPENDENT_VARIABLE == "intensity":
        return "bubble_intensity"
    raise ValueError("DEPENDENT_VARIABLE must be 'duration' or 'intensity'")


def prepare_design_matrix(df):
    """Select the episode outcome and configured regressors, add controls, and remove incomplete rows."""
    y_col = get_dependent_variable_column()
    work = df.copy()

    x_cols = []

    if INCLUDE_PRE_BUBBLE_AVG_VOL:
        x_cols.append("pre_bubble_avg_vol")
    if INCLUDE_VOL_RIGHT_BEFORE_BUBBLE:
        x_cols.append("vol_right_before_bubble")
    if INCLUDE_MARKET_AVG_VOL:
        x_cols.append("market_avg_vol")

    # market type dummies
    if INCLUDE_MARKET_TYPE_DUMMIES:
        market_dummies = pd.get_dummies(
            work["market_type"],
            prefix="market_type",
            drop_first=True,
            dtype=float
        )
        work = pd.concat([work, market_dummies], axis=1)
        x_cols.extend(list(market_dummies.columns))

    # year dummies
    if INCLUDE_YEAR_DUMMIES:
        year_dummies = pd.get_dummies(
            work["bubble_start_year"],
            prefix="year",
            drop_first=True,
            dtype=float
        )
        work = pd.concat([work, year_dummies], axis=1)
        x_cols.extend(list(year_dummies.columns))

    if not x_cols:
        raise ValueError("You turned off all regressors. At least one regressor must be included.")

    model_df = work[[y_col] + x_cols + [
        "source_file", "source_path", "series_name", "market_type",
        "bubble_id", "bubble_start_date", "bubble_end_date",
        "bubble_start_year"
    ]].copy()

    model_df = model_df.dropna(subset=[y_col] + x_cols).reset_index(drop=True)

    if model_df.empty:
        raise ValueError("No usable rows remain after dropping missing values for the selected regressors.")

    return model_df, y_col, x_cols


# =========================================================
# REGRESSION
# =========================================================
def run_regression(model_df, y_col, x_cols):
    """Fit episode-level OLS with HC1 covariance and append fitted values and residuals."""
    y = model_df[y_col].astype(float)
    X = sm.add_constant(model_df[x_cols].astype(float), has_constant="add")

    model = sm.OLS(y, X).fit(cov_type="HC1")

    model_df = model_df.copy()
    model_df["fitted"] = model.predict(X)
    model_df["residual"] = model_df[y_col] - model_df["fitted"]
    model_df["abs_residual"] = np.abs(model_df["residual"])
    model_df["sq_residual"] = model_df["residual"] ** 2

    return model_df, model


# =========================================================
# SAVE OUTPUTS
# =========================================================
def save_outputs_clean(
    output_folder,
    reg_df,
    status_df,
    model_df,
    model,
    x_cols
):
    prefix = f"{DEPENDENT_VARIABLE}_regression"

    reg_data_file = output_folder / f"{prefix}_bubble_level_dataset_full.csv"
    status_file = output_folder / f"{prefix}_file_processing_status.csv"
    used_data_file = output_folder / f"{prefix}_bubble_level_dataset_used.csv"
    coef_file = output_folder / f"{prefix}_coefficients.csv"
    summary_file = output_folder / f"{prefix}_summary.txt"
    config_file = output_folder / f"{prefix}_config.txt"
    residuals_by_market_file = output_folder / f"{prefix}_residuals_by_market_type.csv"
    residuals_by_year_file = output_folder / f"{prefix}_residuals_by_year.csv"
    run_info_file = output_folder / "run_info.txt"

    reg_df.to_csv(reg_data_file, index=False)
    status_df.to_csv(status_file, index=False)
    model_df.to_csv(used_data_file, index=False)

    coef_table = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t_stat": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    coef_table.to_csv(coef_file, index=False)

    residuals_by_market = (
        model_df.groupby("market_type", dropna=False)
        .agg(
            n_obs=("residual", "size"),
            mean_actual=(get_dependent_variable_column(), "mean"),
            mean_fitted=("fitted", "mean"),
            mean_residual=("residual", "mean"),
            mae=("abs_residual", "mean"),
            rmse=("sq_residual", lambda x: np.sqrt(np.mean(x))),
        )
        .reset_index()
    )
    residuals_by_market.to_csv(residuals_by_market_file, index=False)

    residuals_by_year = (
        model_df.groupby("bubble_start_year", dropna=False)
        .agg(
            n_obs=("residual", "size"),
            mean_actual=(get_dependent_variable_column(), "mean"),
            mean_fitted=("fitted", "mean"),
            mean_residual=("residual", "mean"),
            mae=("abs_residual", "mean"),
            rmse=("sq_residual", lambda x: np.sqrt(np.mean(x))),
        )
        .reset_index()
    )
    residuals_by_year.to_csv(residuals_by_year_file, index=False)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    with open(config_file, "w", encoding="utf-8") as f:
        f.write("FINAL BUBBLE REGRESSION CONFIG\n")
        f.write("=" * 80 + "\n")
        f.write(f"DEPENDENT_VARIABLE = {DEPENDENT_VARIABLE}\n")
        f.write(f"INPUT_FOLDERS = {INPUT_FOLDERS}\n")
        f.write(f"SEARCH_RECURSIVELY = {SEARCH_RECURSIVELY}\n")
        f.write(f"PRICE_IS_ALREADY_LOG = {PRICE_IS_ALREADY_LOG}\n")
        f.write(f"ANNUALIZE_VOL = {ANNUALIZE_VOL}\n")
        f.write(f"INCLUDE_PRE_BUBBLE_AVG_VOL = {INCLUDE_PRE_BUBBLE_AVG_VOL}\n")
        f.write(f"INCLUDE_VOL_RIGHT_BEFORE_BUBBLE = {INCLUDE_VOL_RIGHT_BEFORE_BUBBLE}\n")
        f.write(f"INCLUDE_MARKET_AVG_VOL = {INCLUDE_MARKET_AVG_VOL}\n")
        f.write(f"INCLUDE_MARKET_TYPE_DUMMIES = {INCLUDE_MARKET_TYPE_DUMMIES}\n")
        f.write(f"INCLUDE_YEAR_DUMMIES = {INCLUDE_YEAR_DUMMIES}\n")
        f.write(f"PRE_WINDOW_MODE = {PRE_WINDOW_MODE}\n")
        f.write(f"PRE_WINDOW_FIXED = {PRE_WINDOW_FIXED}\n")
        f.write(f"PRE_WINDOW_N_MULTIPLIER = {PRE_WINDOW_N_MULTIPLIER}\n")
        f.write(f"MIN_BUBBLE_DURATION = {MIN_BUBBLE_DURATION}\n")
        f.write(f"MIN_TOTAL_OBS_FOR_FILE = {MIN_TOTAL_OBS_FOR_FILE}\n")
        f.write(f"MIN_RET_OBS_FOR_GARCH = {MIN_RET_OBS_FOR_GARCH}\n")
        f.write(f"Regressors used = {x_cols}\n")
        f.write(f"Observations used = {len(model_df)}\n")
        f.write(f"R-squared = {model.rsquared}\n")
        f.write(f"Adj. R-squared = {model.rsquared_adj}\n")
        f.write("GARCH initial variance = unconditional variance omega / (1 - alpha - beta), with fallback to sample variance if invalid.\n")

    with open(run_info_file, "w", encoding="utf-8") as f:
        f.write(f"Run timestamp: {datetime.now(tz=UTC).astimezone()}\n")
        f.write(f"Dependent variable: {DEPENDENT_VARIABLE}\n")
        f.write(f"Output folder: {output_folder}\n")
        f.write(f"Regressors: {x_cols}\n")
        f.write(f"Observations used: {len(model_df)}\n")

    print("\nSaved files:")
    print(f"  Full dataset               : {reg_data_file}")
    print(f"  File processing status     : {status_file}")
    print(f"  Used dataset               : {used_data_file}")
    print(f"  Coefficients               : {coef_file}")
    print(f"  Summary                    : {summary_file}")
    print(f"  Config                     : {config_file}")
    print(f"  Residuals by market type   : {residuals_by_market_file}")
    print(f"  Residuals by year          : {residuals_by_year_file}")
    print(f"  Run info                   : {run_info_file}")


# =========================================================
# PRINTING
# =========================================================
def print_final_results(model_df, model):
    print("\n" + "=" * 100)
    print("FINAL REGRESSION RESULT")
    print("=" * 100)
    print(model.summary())

    print("\n" + "=" * 100)
    print("USED DATASET PREVIEW")
    print("=" * 100)
    print(model_df.head().to_string(index=False))

    print("\n" + "=" * 100)
    print("RESIDUAL SUMMARY")
    print("=" * 100)
    print(model_df["residual"].describe().to_string())

    print("\n" + "=" * 100)
    print("ABSOLUTE RESIDUALS: LARGEST 20")
    print("=" * 100)
    cols = [
        "source_file", "series_name", "market_type",
        "bubble_id", "bubble_start_date", "bubble_end_date",
        get_dependent_variable_column(), "fitted", "residual", "abs_residual"
    ]
    cols = [c for c in cols if c in model_df.columns]
    print(
        model_df[cols]
        .sort_values("abs_residual", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("SUMMARY BY MARKET TYPE")
    print("=" * 100)
    by_market = (
        model_df.groupby("market_type", dropna=False)
        .agg(
            n_obs=("residual", "size"),
            mean_actual=(get_dependent_variable_column(), "mean"),
            mean_fitted=("fitted", "mean"),
            mean_residual=("residual", "mean"),
            mae=("abs_residual", "mean"),
            rmse=("sq_residual", lambda x: np.sqrt(np.mean(x))),
        )
        .reset_index()
    )
    print(by_market.to_string(index=False))

    print("\n" + "=" * 100)
    print("SUMMARY BY YEAR")
    print("=" * 100)
    by_year = (
        model_df.groupby("bubble_start_year", dropna=False)
        .agg(
            n_obs=("residual", "size"),
            mean_actual=(get_dependent_variable_column(), "mean"),
            mean_fitted=("fitted", "mean"),
            mean_residual=("residual", "mean"),
            mae=("abs_residual", "mean"),
            rmse=("sq_residual", lambda x: np.sqrt(np.mean(x))),
        )
        .reset_index()
    )
    print(by_year.to_string(index=False))


# =========================================================
# MAIN
# =========================================================
def main():
    run_output_folder = create_run_output_folder(BASE_OUTPUT_FOLDER)

    print("\n" + "=" * 80)
    print("OUTPUT FOLDER FOR THIS RUN")
    print("=" * 80)
    print(run_output_folder)

    reg_df, status_df = build_regression_dataset(INPUT_FOLDERS)

    print("\n" + "=" * 100)
    print("BUBBLE-LEVEL DATASET CREATED")
    print("=" * 100)
    print(f"Total bubble observations: {len(reg_df)}")
    print(reg_df.head().to_string(index=False))

    model_df, y_col, x_cols = prepare_design_matrix(reg_df)

    print("\n" + "=" * 100)
    print("MODEL DATASET READY")
    print("=" * 100)
    print(f"Dependent variable: {y_col}")
    print(f"Regressors used   : {x_cols}")
    print(f"Usable observations after NA drop: {len(model_df)}")

    model_df, model = run_regression(model_df, y_col, x_cols)

    print_final_results(model_df, model)

    save_outputs_clean(
        output_folder=run_output_folder,
        reg_df=reg_df,
        status_df=status_df,
        model_df=model_df,
        model=model,
        x_cols=x_cols
    )


if __name__ == "__main__":
    main()
