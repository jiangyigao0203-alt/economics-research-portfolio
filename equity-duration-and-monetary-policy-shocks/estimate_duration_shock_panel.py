"""Estimate equity-return responses to shocks conditional on equity duration.

Merge firm-event abnormal returns, event shocks, lagged monthly duration, and
optional asset betas. MODEL_SPEC selects binned, beta-adjusted, continuous, or
shock-only specifications. Fixed effects, transforms, and covariance options
are configured below. Export estimates, diagnostics, and local analysis panels.
Run from the project directory after reconciling the documented input filenames.
This script performs estimation and writes outputs when imported.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ============================================================
# SETTINGS
# ============================================================
FILE_SHOCKS = "shocks.csv"
FILE_ABRET  = "abnormal_return.csv"
FILE_DUR    = "Implied_Duration_g=0.6.csv"

# ---------- Asset beta settings ----------
ASSET_BETA = True
FILE_ASSET_BETA = "asset_betas.csv"   # existing precomputed beta file
ASSET_BETA_COL_CANDIDATES = ["asset_beta", "beta"]
# ----------------------------------------

OUTPUT_DIR = "panel_outputs_model_selectable"

# Choose which model to run:
#   "binned"           = duration-quantile/bin interaction model
#   "binned_withbeta"  = binned model + beta*shock interaction terms
#   "continuous"       = continuous duration interaction model
#   "shock_only"       = pooled event-shock model (benchmark; no duration heterogeneity)
MODEL_SPEC = "binned_withbeta"

# If MODEL_SPEC == "binned" or "binned_withbeta"
BINNING_MODE = "quantile"   # "quantile" or "integer_round"
N_QUANTILES = 20             # used only if BINNING_MODE == "quantile"
BIN_MIN = -100               # used only if BINNING_MODE == "integer_round"
BIN_MAX = 200                # used only if BINNING_MODE == "integer_round"
REFERENCE_BIN = 1            # for quantile mode, usually 1; for integer mode, often 0

# Duration transform options for models using duration
USE_DURATION_TRANSFORM = "level"   # "level", "log", "asinh", "zscore"
WINSORIZE_DURATION = True
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

# Fixed effects
INCLUDE_FIRM_FE = True
INCLUDE_EVENT_FE = False

# Covariance choice
COV_TYPE = "HC1"            # e.g. "HC1"

# Numerical settings
TWOWAY_TOL = 1e-10
TWOWAY_MAX_ITER = 200
IDENTIFICATION_TOL = 1e-14

MATCH_PROGRESS_EVERY_FIRMS = 250
RESID_PROGRESS_EVERY_REGRESSORS = 25
TERM_PROGRESS_EVERY_BINS = 25

# ============================================================
# OUTPUT FILES
# ============================================================
OUTPUT_REG_DATA              = "panel_regression_ready.csv"
OUTPUT_COEFS                 = "panel_regression_coefficients.csv"
OUTPUT_PROGRESS_LOG          = "panel_regression_progress_log.txt"
OUTPUT_SUMMARY_TXT           = "panel_regression_summary.txt"
OUTPUT_MODEL_STATS_CSV       = "panel_regression_model_stats.csv"
OUTPUT_SAMPLE_STATS_CSV      = "panel_regression_sample_stats.csv"
OUTPUT_EVENT_STATS_CSV       = "panel_regression_event_stats.csv"
OUTPUT_BIN_STATS_CSV         = "panel_regression_bin_stats.csv"
OUTPUT_INPUT_AUDIT_CSV       = "panel_regression_input_audit.csv"
OUTPUT_MATCH_QUALITY_CSV     = "panel_regression_match_quality.csv"
OUTPUT_INTERACTION_STATS_CSV = "panel_regression_interaction_stats.csv"
OUTPUT_PANEL_PREVIEW_CSV     = "panel_regression_panel_preview.csv"
OUTPUT_MERGE_DIAG_CSV        = "panel_regression_merge_diagnostics.csv"
OUTPUT_RESIDUALIZED_CSV      = "panel_regression_residualized_data.csv"
OUTPUT_IDENTIFICATION_CSV    = "panel_regression_identification_check.csv"
OUTPUT_ASSET_BETA_MERGE_CSV  = "panel_regression_asset_beta_merge.csv"
OUTPUT_STATS_TABLE_HTML      = "panel_stats_summary_pretty.html"

# ============================================================
# COLUMN SETTINGS
# ============================================================
GVKEY_COL = "gvkey"
SHOCK_EVENT_DATE_COL = "date_time"
MP_COL = "eps_MP"
INFO_COL = "eps_INFO"
ABRET_EVENT_DATE_COL = "event_date"
ABRET_DEP_CANDIDATES = [
    "CAR_0_4", "CAR_0_9", "CAR_0_14", "CAR_0_19",
    "car_0_4", "car_0_9", "car_0_14", "car_0_19",
    "CAR", "car", "abnormal_return"
]
DUR_DATE_COL = "month_date"
DUR_VALUE_CANDIDATES = ["implied_duration_years", "duration", "dur"]

# ============================================================
# PATH SETUP
# ============================================================
OUTDIR = Path(OUTPUT_DIR)
OUTDIR.mkdir(parents=True, exist_ok=True)

def outpath(filename: str) -> Path:
    return OUTDIR / filename

# ============================================================
# LOGGING
# ============================================================
PROGRESS_LOG = []

def log_progress(message: str):
    line = f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    PROGRESS_LOG.append(line)
    print(line, flush=True)

# ============================================================
# HELPERS
# ============================================================

def pick_first_existing(columns, candidates, label):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"Could not find {label}. Tried {candidates}")


def load_asset_beta_file(file_path, gvkey_col="gvkey"):
    """Read firm identifiers and the first supported asset-beta column from a CSV."""
    beta_df = pd.read_csv(file_path)

    if gvkey_col not in beta_df.columns:
        raise ValueError(f"Missing {gvkey_col} in asset beta file: {file_path}")

    beta_col = pick_first_existing(
        beta_df.columns,
        ASSET_BETA_COL_CANDIDATES,
        "asset beta column",
    )

    beta_df = beta_df[[gvkey_col, beta_col]].copy()
    beta_df[gvkey_col] = beta_df[gvkey_col].astype(str).str.zfill(6)
    beta_df["asset_beta"] = pd.to_numeric(beta_df[beta_col], errors="coerce")
    beta_df = beta_df.drop(columns=[beta_col]).drop_duplicates(subset=[gvkey_col])

    return beta_df


def winsorize_series(s, lower=0.01, upper=0.99):
    """Clip a series at its lower and upper empirical quantiles."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def series_stats(s, prefix):
    s = pd.to_numeric(s, errors="coerce")
    return {
        f"{prefix}_count": int(s.notna().sum()),
        f"{prefix}_mean": float(s.mean()) if s.notna().any() else np.nan,
        f"{prefix}_std": float(s.std()) if s.notna().sum() > 1 else np.nan,
        f"{prefix}_min": float(s.min()) if s.notna().any() else np.nan,
        f"{prefix}_median": float(s.median()) if s.notna().any() else np.nan,
        f"{prefix}_max": float(s.max()) if s.notna().any() else np.nan,
    }


def transform_duration(s, mode):
    """Apply the selected level, log, inverse-hyperbolic-sine, or standardized duration transform."""
    s = pd.to_numeric(s, errors="coerce")
    if mode == "level":
        return s
    if mode == "log":
        if (s <= 0).any():
            bad = int((s <= 0).sum())
            raise ValueError(f"log transform requested but found {bad:,} duration values <= 0.")
        return np.log(s)
    if mode == "asinh":
        return np.arcsinh(s)
    if mode == "zscore":
        mean = s.mean()
        std = s.std()
        if pd.isna(std) or abs(std) < 1e-14:
            return pd.Series(np.nan, index=s.index)
        return (s - mean) / std
    raise ValueError(f"Unknown USE_DURATION_TRANSFORM option: {mode}")


def duration_to_integer_bin(s):
    return np.floor(pd.to_numeric(s, errors="coerce") + 0.5).astype("Int64")


def assign_quantile_bins(s, n_quantiles):
    """Assign duration quantile groups, allowing duplicate cutpoints to reduce the bin count."""
    x = pd.to_numeric(s, errors="coerce")
    valid = x.notna()
    out = pd.Series(pd.NA, index=x.index, dtype="Int64")
    if valid.sum() == 0:
        return out

    ranked = x[valid].rank(method="first")
    try:
        q = pd.qcut(ranked, q=n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        unique_count = x[valid].nunique()
        q = pd.qcut(ranked, q=min(n_quantiles, unique_count), labels=False, duplicates="drop")

    out.loc[valid] = pd.Series(q, index=ranked.index).astype("Int64") + 1
    return out


def two_way_demean(df, col, entity_col="gvkey", time_col="event_date", tol=1e-10, max_iter=200):
    """Iteratively remove entity and time means until convergence or the iteration limit."""
    z = pd.to_numeric(df[col], errors="coerce").astype(float).copy()
    for _ in range(max_iter):
        z_old = z.copy()
        z = z - z.groupby(df[entity_col]).transform("mean")
        z = z - z.groupby(df[time_col]).transform("mean")
        diff = np.nanmax(np.abs(z - z_old))
        if pd.isna(diff) or diff < tol:
            break
    return z


def one_way_demean(df, col, group_col):
    """Remove group means from one column while retaining observation alignment."""
    z = pd.to_numeric(df[col], errors="coerce").astype(float).copy()
    return z - z.groupby(df[group_col]).transform("mean")


def residualize_column(df, col, entity_col="gvkey", time_col="event_date"):
    """Remove the configured fixed effects from one regression variable."""
    if INCLUDE_FIRM_FE and INCLUDE_EVENT_FE:
        return two_way_demean(
            df, col,
            entity_col=entity_col,
            time_col=time_col,
            tol=TWOWAY_TOL,
            max_iter=TWOWAY_MAX_ITER,
        )
    if INCLUDE_FIRM_FE:
        return one_way_demean(df, col, entity_col)
    if INCLUDE_EVENT_FE:
        return one_way_demean(df, col, time_col)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def residualize_panel(df, y_col, x_cols, entity_col="gvkey", time_col="event_date", log_func=None):
    """Apply the configured fixed-effect transformation to the outcome and regressor columns."""
    if log_func:
        log_func("Residualizing dependent variable with respect to requested fixed effects.")

    out_cols = {
        y_col: residualize_column(df, y_col, entity_col=entity_col, time_col=time_col)
    }

    n = len(x_cols)
    for j, col in enumerate(x_cols, start=1):
        if log_func and (j == 1 or j % RESID_PROGRESS_EVERY_REGRESSORS == 0 or j == n):
            log_func(f"Residualizing regressor {j:,}/{n:,}: {col}")
        out_cols[col] = residualize_column(df, col, entity_col=entity_col, time_col=time_col)

    out = pd.concat({k: pd.Series(v, index=df.index) for k, v in out_cols.items()}, axis=1)
    out.columns = list(out_cols.keys())
    return out.copy()


def fmt_num(x, digits=3, pct=False):
    if pd.isna(x):
        return ""
    if pct:
        return f"{100 * float(x):.{digits}f}%"
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def safe_metric_lookup(df, metric, default: float | str = np.nan):
    if df is None or len(df) == 0 or "metric" not in df.columns or "value" not in df.columns:
        return default
    m = df.loc[df["metric"] == metric, "value"]
    if len(m) == 0:
        return default
    return m.iloc[0]


def build_panel_b_like_statsmodels(coef_table):
    panel_b = coef_table.copy()

    rename_map = {
        "term": "Duration portfolio",
        "coef": "coef",
        "std_err": "std err",
        "z_stat": "z",
        "p_value": "P>|z|",
    }
    keep_cols = [c for c in rename_map if c in panel_b.columns]
    panel_b = panel_b[keep_cols].rename(columns=rename_map).copy()

    if panel_b.empty:
        return panel_b

    wanted = (
        panel_b["Duration portfolio"].astype(str).str.startswith(("MP_q", "INFO_q", "BETA_"))
        | panel_b["Duration portfolio"].astype(str).isin([MP_COL, INFO_COL])
    )
    panel_b = panel_b.loc[wanted].copy()

    def sort_key(term):
        term = str(term)
        if term == MP_COL:
            return (0, 0)
        if term == INFO_COL:
            return (1, 0)
        if term.startswith("MP_q"):
            try:
                return (2, int(term.replace("MP_q", "")))
            except ValueError:
                return (2, 999999)
        if term.startswith("INFO_q"):
            try:
                return (3, int(term.replace("INFO_q", "")))
            except ValueError:
                return (3, 999999)
        if term == "BETA_MP":
            return (4, 0)
        if term == "BETA_INFO":
            return (4, 1)
        return (9, 999999)

    panel_b = panel_b.sort_values(
        by="Duration portfolio",
        key=lambda s: s.map(sort_key)
    ).reset_index(drop=True)

    for c in ["coef", "std err", "z", "P>|z|"]:
        if c in panel_b.columns:
            panel_b[c] = panel_b[c].apply(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")

    return panel_b


def build_pretty_stats_html(panel_bin_stats, sample_stats_rows, model_stats, model_spec, reference_bin, coef_table):
    sample_stats_df = pd.DataFrame(sample_stats_rows).copy()

    panel_a = pd.DataFrame([
        {"Metric": "Observations", "Value": fmt_num(safe_metric_lookup(sample_stats_df, "n_firm_event_obs"), 0)},
        {"Metric": "Firms", "Value": fmt_num(safe_metric_lookup(sample_stats_df, "n_firms"), 0)},
        {"Metric": "Events", "Value": fmt_num(safe_metric_lookup(sample_stats_df, "n_events"), 0)},
        {"Metric": "Dep. var mean", "Value": fmt_num(safe_metric_lookup(sample_stats_df, "dep_var_mean"), 4)},
        {"Metric": "Dep. var std", "Value": fmt_num(safe_metric_lookup(sample_stats_df, "dep_var_std"), 4)},
        {"Metric": "Duration mean", "Value": fmt_num(safe_metric_lookup(sample_stats_df, "duration_transformed_mean"), 4)},
        {"Metric": "Duration median", "Value": fmt_num(safe_metric_lookup(sample_stats_df, "duration_transformed_median"), 4)},
        {"Metric": "R²", "Value": fmt_num(safe_metric_lookup(model_stats, "rsquared_uncentered"), 4)},
        {"Metric": "Adj. R²", "Value": fmt_num(safe_metric_lookup(model_stats, "rsquared_adj_uncentered"), 4)},
        {"Metric": "Covariance", "Value": str(safe_metric_lookup(model_stats, "cov_type", ""))},
        {"Metric": "Model", "Value": str(model_spec)},
        {"Metric": "Reference bin", "Value": "" if reference_bin is None else str(reference_bin)},
    ])

    panel_b = build_panel_b_like_statsmodels(coef_table)

    panel_a_html = panel_a.to_html(index=False, escape=False, border=0)
    panel_b_html = panel_b.to_html(index=False, escape=False, border=0)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Panel Statistics Summary</title>
<style>
    body {{
        font-family: "Times New Roman", Georgia, serif;
        margin: 28px 36px;
        color: #111;
        background: #fff;
    }}
    .title {{
        text-align: center;
        font-size: 20px;
        margin-bottom: 12px;
    }}
    .subtitle {{
        text-align: center;
        font-size: 14px;
        margin-bottom: 24px;
    }}
    .panel-title {{
        text-align: center;
        font-size: 17px;
        margin-top: 18px;
        margin-bottom: 8px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 0 auto 16px auto;
        font-size: 14px;
    }}
    thead th {{
        border-top: 2px solid #222;
        border-bottom: 1.5px solid #222;
        padding: 6px 10px;
        text-align: center;
        font-weight: normal;
    }}
    tbody td {{
        padding: 5px 10px;
        vertical-align: middle;
        white-space: nowrap;
    }}
    table.panel-a tbody td {{
        text-align: center;
    }}
    table.panel-a tbody td:first-child {{
        text-align: left;
    }}
    table.panel-b tbody td {{
        text-align: right;
    }}
    table.panel-b tbody td:first-child {{
        text-align: left;
    }}
    tbody tr:last-child td {{
        border-bottom: 2px solid #222;
    }}
    .notes {{
        margin-top: 14px;
        font-size: 12px;
        line-height: 1.4;
    }}
</style>
</head>
<body>
    <div class="title">Panel Statistics Summary</div>
    <div class="subtitle">Paper-style table formatted for easy copy/paste into Google Docs</div>

    <div class="panel-title">Panel A: Overall sample summary</div>
    <div class="panel-a">{panel_a_html}</div>

    <div class="panel-title">Panel B: Coefficient Estimates</div>
    <div class="panel-b">{panel_b_html}</div>

    <div class="notes">
        Notes: Panel A reports overall sample and model summary statistics. Panel B reports coefficient estimates in a statsmodels-style block for direct pasting into Google Docs.
    </div>
</body>
</html>
"""
    html = html.replace('<table ', '<table class="panel-a" ', 1)
    html = html.replace('<table ', '<table class="panel-b" ', 1)
    return html


def model_description(model_spec, reference_bin=None):
    fe_text = []
    if INCLUDE_FIRM_FE:
        fe_text.append("firm FE")
    if INCLUDE_EVENT_FE:
        fe_text.append("event FE")
    fe_text = " + ".join(fe_text) if fe_text else "no FE"

    if model_spec == "binned":
        beta_text = " + theta*asset_beta" if ASSET_BETA else ""
        return (
            "CAR_it = beta_ref_MP*eps_MP_t + beta_ref_INFO*eps_INFO_t + "
            "sum_{q != ref} beta_q*(D_q(duration_it-) * eps_MP_t) + "
            "sum_{q != ref} gamma_q*(D_q(duration_it-) * eps_INFO_t)"
            f"{beta_text} + {fe_text} + u_it | omitted reference bin = {reference_bin}"
        )
    if model_spec == "binned_withbeta":
        return (
            "CAR_it = beta_ref_MP*eps_MP_t + beta_ref_INFO*eps_INFO_t + "
            "sum_{q != ref} beta_q*(D_q(duration_it-) * eps_MP_t) + "
            "sum_{q != ref} gamma_q*(D_q(duration_it-) * eps_INFO_t) + "
            "theta_MP*(asset_beta_i * eps_MP_t) + theta_INFO*(asset_beta_i * eps_INFO_t) + "
            f"{fe_text} + u_it | omitted reference bin = {reference_bin}"
        )
    if model_spec == "continuous":
        return (
            "CAR_it = beta_MP*(duration_it- * eps_MP_t) + "
            "beta_INFO*(duration_it- * eps_INFO_t) + "
            f"{fe_text} + u_it"
        )
    if model_spec == "shock_only":
        return (
            "CAR_it = beta_MP*eps_MP_t + beta_INFO*eps_INFO_t + "
            f"{fe_text} + u_it"
        )
    return "Unknown model"

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
log_progress("STEP 1: Loading input files.")
shocks = pd.read_csv(FILE_SHOCKS)
abret = pd.read_csv(FILE_ABRET)
dur = pd.read_csv(FILE_DUR)

input_audit_rows = [
    {"metric": "shocks_rows_raw", "value": len(shocks)},
    {"metric": "abret_rows_raw", "value": len(abret)},
    {"metric": "dur_rows_raw", "value": len(dur)},
    {"metric": "model_spec", "value": MODEL_SPEC},
    {"metric": "include_firm_fe", "value": INCLUDE_FIRM_FE},
    {"metric": "include_event_fe", "value": INCLUDE_EVENT_FE},
    {"metric": "asset_beta_enabled", "value": ASSET_BETA},
]

for name, df in [("abret", abret), ("dur", dur)]:
    if GVKEY_COL not in df.columns:
        raise ValueError(f"Missing {GVKEY_COL} in {name} file.")
    df[GVKEY_COL] = df[GVKEY_COL].astype(str).str.zfill(6)

DEP_COL = pick_first_existing(abret.columns, ABRET_DEP_CANDIDATES, "dependent variable column")
DUR_VALUE_COL = pick_first_existing(dur.columns, DUR_VALUE_CANDIDATES, "duration column")

log_progress(f"Detected dependent variable column: {DEP_COL}")
log_progress(f"Detected duration column: {DUR_VALUE_COL}")
log_progress(f"Model selected: {MODEL_SPEC}")
log_progress(f"Duration transform setting: {USE_DURATION_TRANSFORM}")
log_progress(f"Asset beta enabled: {ASSET_BETA}")
log_progress(f"Output directory: {OUTDIR.resolve()}")

shocks[SHOCK_EVENT_DATE_COL] = pd.to_datetime(shocks[SHOCK_EVENT_DATE_COL], errors="coerce").dt.normalize()
abret[ABRET_EVENT_DATE_COL] = pd.to_datetime(abret[ABRET_EVENT_DATE_COL], errors="coerce").dt.normalize()
dur[DUR_DATE_COL] = pd.to_datetime(dur[DUR_DATE_COL], errors="coerce").dt.normalize()

shocks = shocks.dropna(subset=[SHOCK_EVENT_DATE_COL, MP_COL, INFO_COL]).copy()
abret = abret.dropna(subset=[GVKEY_COL, ABRET_EVENT_DATE_COL, DEP_COL]).copy()
dur = dur.dropna(subset=[GVKEY_COL, DUR_DATE_COL, DUR_VALUE_COL]).copy()

input_audit_rows += [
    {"metric": "shocks_rows_after_dropna", "value": len(shocks)},
    {"metric": "abret_rows_after_dropna", "value": len(abret)},
    {"metric": "dur_rows_after_dropna", "value": len(dur)},
]

# ============================================================
# STEP 2: BUILD FIRM-EVENT DEPENDENT VARIABLE
# ============================================================
log_progress("STEP 2: Building firm-event dependent variable.")
if DEP_COL.lower() in {"car_0_4", "car_0_9", "car_0_14", "car_0_19", "car"}:
    dep = (
        abret[[GVKEY_COL, ABRET_EVENT_DATE_COL, DEP_COL]]
        .rename(columns={ABRET_EVENT_DATE_COL: "event_date", DEP_COL: "dep_var"})
        .dropna(subset=["dep_var"])
        .drop_duplicates(subset=[GVKEY_COL, "event_date"])
        .copy()
    )
    dep_construction = f"Used existing {DEP_COL} directly as dep_var."
else:
    dep = (
        abret[[GVKEY_COL, ABRET_EVENT_DATE_COL, DEP_COL]]
        .rename(columns={ABRET_EVENT_DATE_COL: "event_date", DEP_COL: "abret_component"})
        .groupby([GVKEY_COL, "event_date"], as_index=False)[["abret_component"]]
        .sum()
        .rename(columns={"abret_component": "dep_var"})
    )
    dep_construction = f"Constructed dep_var by summing {DEP_COL} within each (gvkey, event_date)."
log_progress(f"Firm-event dependent-variable rows created: {len(dep):,}")

# ============================================================
# STEP 3: PREPARE DURATION PANEL
# ============================================================
log_progress("STEP 3: Preparing duration panel.")
dur = (
    dur[[GVKEY_COL, DUR_DATE_COL, DUR_VALUE_COL]]
    .rename(columns={DUR_DATE_COL: "dur_date", DUR_VALUE_COL: "duration_value_raw"})
    .sort_values([GVKEY_COL, "dur_date"])
    .reset_index(drop=True)
)

# ============================================================
# STEP 4: MERGE DEPENDENT VARIABLE WITH SHOCKS
# ============================================================
log_progress("STEP 4: Merging firm-event data with shocks.")
shock_key = shocks[[SHOCK_EVENT_DATE_COL, MP_COL, INFO_COL]].rename(columns={SHOCK_EVENT_DATE_COL: "event_date"}).copy()
if shock_key["event_date"].duplicated().any():
    dup_count = int(shock_key["event_date"].duplicated().sum())
    log_progress(f"Detected {dup_count:,} duplicate shock-date rows. Collapsing duplicates.")
    shock_key = (
        shock_key.sort_values("event_date")
        .groupby("event_date", as_index=False)
        .agg({MP_COL: "first", INFO_COL: "first"})
    )

shock_date_set = set(shock_key["event_date"].dropna().unique())
dep_date_set = set(dep["event_date"].dropna().unique())
merge_diag = pd.DataFrame([
    {"metric": "unique_dep_event_dates", "value": len(dep_date_set)},
    {"metric": "unique_shock_event_dates", "value": len(shock_date_set)},
    {"metric": "overlapping_event_dates", "value": len(dep_date_set.intersection(shock_date_set))},
])
merge_diag.to_csv(outpath(OUTPUT_MERGE_DIAG_CSV), index=False)
log_progress(f"Saved merge diagnostics to {outpath(OUTPUT_MERGE_DIAG_CSV)}")

panel = dep.merge(shock_key, on="event_date", how="inner")
rows_after_shock_merge = len(panel)
log_progress(f"Rows after merging with shocks: {rows_after_shock_merge:,}")
if rows_after_shock_merge == 0:
    raise ValueError("No rows matched between dependent-variable event_date and shocks event_date.")

# ============================================================
# STEP 5: MATCH LAGGED DURATION
# ============================================================
needs_duration = MODEL_SPEC in {"binned", "binned_withbeta", "continuous"}
match_quality_rows = [
    {"metric": "rows_after_shock_merge", "value": int(rows_after_shock_merge)}
]

if needs_duration:
    log_progress("STEP 5: Matching the latest pre-event duration to each firm-event.")
    panel_for_merge = panel.sort_values([GVKEY_COL, "event_date"]).copy()
    dur_for_merge = dur.sort_values([GVKEY_COL, "dur_date"]).copy()
    dur_groups = {k: g.copy() for k, g in dur_for_merge.groupby(GVKEY_COL)}
    matched_parts = []
    start_match = time.time()
    panel_groups = list(panel_for_merge.groupby(GVKEY_COL))
    total_groups = len(panel_groups)

    for i, (gvkey, g_panel) in enumerate(panel_groups, start=1):
        if i == 1 or i % MATCH_PROGRESS_EVERY_FIRMS == 0 or i == total_groups:
            elapsed = time.time() - start_match
            rate = i / elapsed if elapsed > 0 else np.nan
            eta = (total_groups - i) / rate if rate and rate > 0 else np.nan
            eta_str = f"{eta/60:.1f} min" if np.isfinite(eta) else "n/a"
            log_progress(
                f"Duration matching progress: processed {i:,}/{total_groups:,} firms | "
                f"elapsed={elapsed:.1f}s | ETA={eta_str}"
            )

        g_dur = dur_groups.get(gvkey)
        if g_dur is None or g_dur.empty:
            continue

        tmp = pd.merge_asof(
            g_panel.sort_values("event_date"),
            g_dur.sort_values("dur_date"),
            left_on="event_date",
            right_on="dur_date",
            by=GVKEY_COL,
            direction="backward",
            allow_exact_matches=False,
        )
        matched_parts.append(tmp)

    if len(matched_parts) == 0:
        raise ValueError("No firm-event rows matched to lagged duration.")

    panel = pd.concat(matched_parts, ignore_index=True)
    rows_before_duration_drop = len(panel)
    panel = panel.dropna(subset=["duration_value_raw"]).copy()
    rows_after_duration_drop = len(panel)
    if rows_after_duration_drop == 0:
        raise ValueError("All rows were lost after lagged-duration matching.")

    panel["days_since_duration_measure"] = (panel["event_date"] - panel["dur_date"]).dt.days
    match_quality_rows += [
        {"metric": "rows_before_drop_missing_duration", "value": int(rows_before_duration_drop)},
        {"metric": "rows_after_drop_missing_duration", "value": int(rows_after_duration_drop)},
    ]
    for k, v in series_stats(panel["days_since_duration_measure"], "days_since_duration_measure").items():
        match_quality_rows.append({"metric": k, "value": v})

    log_progress(f"Rows after lagged-duration matching: {len(panel):,}")
else:
    log_progress("STEP 5: Skipping duration matching because MODEL_SPEC = shock_only.")
    panel["dur_date"] = pd.NaT
    panel["duration_value_raw"] = np.nan
    panel["days_since_duration_measure"] = np.nan

# ============================================================
# STEP 5B: LOAD / MERGE EXISTING ASSET BETA
# ============================================================
if ASSET_BETA or MODEL_SPEC == "binned_withbeta":
    log_progress("STEP 5B: Loading existing asset beta file.")
    beta_df = load_asset_beta_file(FILE_ASSET_BETA, gvkey_col=GVKEY_COL)

    before_beta_merge = len(panel)
    panel = panel.merge(beta_df, on=GVKEY_COL, how="left")

    if len(panel) != before_beta_merge:
        raise ValueError("Row count changed after asset beta merge, which should not happen.")

    matched_beta = int(panel["asset_beta"].notna().sum())
    missing_beta_rows = int(panel["asset_beta"].isna().sum())
    log_progress(
        f"Merged existing asset beta into panel. Non-missing beta rows: {matched_beta:,}/{len(panel):,}; "
        f"missing beta rows: {missing_beta_rows:,}"
    )

    beta_merge_stats = (
        panel[[GVKEY_COL, "asset_beta"]]
        .drop_duplicates(subset=[GVKEY_COL])
        .assign(beta_matched=lambda x: x["asset_beta"].notna().astype(int))
    )
    beta_merge_stats.to_csv(outpath(OUTPUT_ASSET_BETA_MERGE_CSV), index=False)
    log_progress(f"Saved asset beta merge diagnostics to {outpath(OUTPUT_ASSET_BETA_MERGE_CSV)}")

    input_audit_rows += [
        {"metric": "asset_beta_file", "value": FILE_ASSET_BETA},
        {"metric": "asset_beta_nonmissing_rows", "value": matched_beta},
        {"metric": "asset_beta_unique_firms_matched", "value": int(beta_merge_stats["beta_matched"].sum())},
    ]
else:
    input_audit_rows += [
        {"metric": "asset_beta_file", "value": ""},
        {"metric": "asset_beta_nonmissing_rows", "value": 0},
        {"metric": "asset_beta_unique_firms_matched", "value": 0},
    ]

# ============================================================
# STEP 6: PREPARE MODEL VARIABLES
# ============================================================
log_progress("STEP 6: Preparing model variables.")
present_bins = []
kept_bins = []
reference_bin = None
requested_x_cols = []

if MODEL_SPEC in {"binned", "binned_withbeta", "continuous"}:
    panel["duration_value"] = panel["duration_value_raw"]
    panel["duration_for_model"] = (
        winsorize_series(panel["duration_value"], WINSOR_LOWER, WINSOR_UPPER)
        if WINSORIZE_DURATION else panel["duration_value"]
    )
    if WINSORIZE_DURATION:
        log_progress("Applied global winsorization to duration_for_model.")

    panel["duration_transformed"] = transform_duration(panel["duration_for_model"], USE_DURATION_TRANSFORM)
    if panel["duration_transformed"].isna().any():
        missing = int(panel["duration_transformed"].isna().sum())
        log_progress(f"Dropping {missing:,} rows with missing transformed duration.")
        panel = panel.dropna(subset=["duration_transformed"]).copy()

if MODEL_SPEC in {"binned", "binned_withbeta"}:
    if BINNING_MODE == "quantile":
        panel["duration_bin"] = assign_quantile_bins(panel["duration_transformed"], N_QUANTILES)
    elif BINNING_MODE == "integer_round":
        panel["duration_bin"] = duration_to_integer_bin(panel["duration_transformed"])
        panel = panel[panel["duration_bin"].between(BIN_MIN, BIN_MAX, inclusive="both")].copy()
    else:
        raise ValueError("BINNING_MODE must be 'quantile' or 'integer_round'.")

    if len(panel) == 0:
        raise ValueError("All rows were lost after assigning duration bins.")

    present_bins = [int(q) for q in panel["duration_bin"].dropna().sort_values().unique().tolist()]
    if not present_bins:
        raise ValueError("No occupied duration bins remain in the sample.")

    reference_bin = REFERENCE_BIN if REFERENCE_BIN in present_bins else present_bins[0]
    if reference_bin != REFERENCE_BIN:
        log_progress(f"Reference bin {REFERENCE_BIN} not present. Using {reference_bin} instead.")

    log_progress(
        f"Duration binning complete: kept {len(present_bins):,} occupied bins; "
        f"reference bin = {reference_bin}."
    )

    log_progress("STEP 7: Creating duration-bin shock interaction terms.")
    kept_bins = [q for q in present_bins if q != reference_bin]

    if not kept_bins:
        log_progress("Only the reference bin is occupied. No non-reference duration bins to create.")
    start_terms = time.time()
    interaction_cols = {}

    for j, q in enumerate(kept_bins, start=1):
        mask = (panel["duration_bin"] == q).to_numpy(dtype=np.int8)
        mpcol = f"MP_q{q}"
        infocol = f"INFO_q{q}"
        interaction_cols[mpcol] = mask * panel[MP_COL].to_numpy()
        interaction_cols[infocol] = mask * panel[INFO_COL].to_numpy()
        requested_x_cols.extend([mpcol, infocol])

        if j == 1 or j % TERM_PROGRESS_EVERY_BINS == 0 or j == len(kept_bins):
            elapsed = time.time() - start_terms
            log_progress(
                f"Interaction-term progress: processed {j:,}/{len(kept_bins):,} bins | elapsed={elapsed:.1f}s"
            )

    if len(interaction_cols) > 0:
        interaction_df = pd.DataFrame(interaction_cols, index=panel.index)
        panel = pd.concat([panel, interaction_df], axis=1).copy()
        log_progress(
            f"Added {interaction_df.shape[1]:,} interaction columns to panel in one concat step to avoid fragmentation."
        )

    # NEW: add the raw shock regressors themselves so the reference bin is the baseline group
    requested_x_cols.extend([MP_COL, INFO_COL])
    log_progress(f"Added baseline shock regressors using original columns: {MP_COL}, {INFO_COL}")

    if MODEL_SPEC == "binned" and ASSET_BETA:
        if "asset_beta" not in panel.columns:
            raise ValueError("ASSET_BETA=True but asset_beta was not merged into panel.")
        panel["asset_beta"] = pd.to_numeric(panel["asset_beta"], errors="coerce").fillna(0.0)
        requested_x_cols.append("asset_beta")
        log_progress("Added asset_beta to requested regressors for binned model, with missing beta filled as 0.")

    if MODEL_SPEC == "binned_withbeta":
        if "asset_beta" not in panel.columns:
            raise ValueError("MODEL_SPEC='binned_withbeta' requires asset_beta to be merged into panel.")
        panel["asset_beta"] = pd.to_numeric(panel["asset_beta"], errors="coerce").fillna(0.0)
        panel["BETA_MP"] = panel["asset_beta"] * panel[MP_COL]
        panel["BETA_INFO"] = panel["asset_beta"] * panel[INFO_COL]
        requested_x_cols.extend(["BETA_MP", "BETA_INFO"])
        log_progress("Added BETA_MP and BETA_INFO regressors for binned_withbeta model, with missing beta filled as 0.")

elif MODEL_SPEC == "continuous":
    panel["MP_dur"] = panel["duration_transformed"] * panel[MP_COL]
    panel["INFO_dur"] = panel["duration_transformed"] * panel[INFO_COL]
    requested_x_cols = ["MP_dur", "INFO_dur"]
    panel["duration_bin"] = pd.NA
    log_progress("Created continuous interaction terms: MP_dur, INFO_dur")

elif MODEL_SPEC == "shock_only":
    panel["MP_shock"] = panel[MP_COL]
    panel["INFO_shock"] = panel[INFO_COL]
    requested_x_cols = ["MP_shock", "INFO_shock"]
    panel["duration_value"] = np.nan
    panel["duration_for_model"] = np.nan
    panel["duration_transformed"] = np.nan
    panel["duration_bin"] = pd.NA
    log_progress("Created pooled shock-only regressors: MP_shock, INFO_shock")

else:
    raise ValueError("MODEL_SPEC must be one of: 'binned', 'binned_withbeta', 'continuous', 'shock_only'.")

# ============================================================
# STEP 8: ESTIMATE MODEL
# ============================================================
log_progress("STEP 8: Estimating panel model.")
print("\nRequested model:\n")
print(model_description(MODEL_SPEC, reference_bin=reference_bin))

reg_df = panel[[GVKEY_COL, "event_date", "dep_var"] + requested_x_cols].dropna().copy()
log_progress(f"Regression rows after dropping missing values: {len(reg_df):,}")
if len(reg_df) == 0:
    raise ValueError("No usable regression observations after dropping missing values.")

resid_df = residualize_panel(
    reg_df,
    y_col="dep_var",
    x_cols=requested_x_cols,
    entity_col=GVKEY_COL,
    time_col="event_date",
    log_func=log_progress,
)

ident_rows = []
kept_x_cols = []
dropped_x_cols = []
for col in requested_x_cols:
    s = pd.to_numeric(resid_df[col], errors="coerce")
    var = float(np.nanvar(s))
    max_abs = float(np.nanmax(np.abs(s))) if s.notna().any() else np.nan
    identified = bool(
        (not pd.isna(var)) and var > IDENTIFICATION_TOL and
        (not pd.isna(max_abs)) and max_abs > IDENTIFICATION_TOL
    )
    ident_rows.append({
        "regressor": col,
        "post_fe_variance": var,
        "post_fe_max_abs": max_abs,
        "identified_after_fe": identified,
    })
    if identified:
        kept_x_cols.append(col)
    else:
        dropped_x_cols.append(col)

pd.DataFrame(ident_rows).to_csv(outpath(OUTPUT_IDENTIFICATION_CSV), index=False)
log_progress(f"Saved identification check to {outpath(OUTPUT_IDENTIFICATION_CSV)}")

if not kept_x_cols:
    raise ValueError(
        "No regressors remain after FE residualization. "
        "This occurs, for example, with shock_only and event fixed effects, "
        "or with time-invariant asset_beta and firm fixed effects. "
        "Turn off INCLUDE_EVENT_FE for the pooled shock-only benchmark, "
        "or disable INCLUDE_FIRM_FE to estimate the plain asset_beta main effect."
    )

resid_df[["dep_var"] + kept_x_cols].to_csv(outpath(OUTPUT_RESIDUALIZED_CSV), index=False)
log_progress(f"Saved residualized regression data to {outpath(OUTPUT_RESIDUALIZED_CSV)}")

Y_tilde = resid_df["dep_var"].astype(float)
X_tilde = resid_df[kept_x_cols].astype(float)
model = sm.OLS(Y_tilde, X_tilde)
results = model.fit(cov_type=COV_TYPE)
log_progress("Regression estimation completed.")

print("\n================ REGRESSION SUMMARY ================\n")
print(results.summary())

# ============================================================
# STEP 9: OUTPUT TABLES
# ============================================================
log_progress("STEP 9: Building output tables.")
coef_table = pd.DataFrame({
    "term": results.params.index,
    "coef": results.params.values,
    "std_err": results.bse.values,
    "z_stat": results.tvalues.values,
    "p_value": results.pvalues.values,
    "ci_lower_95": results.conf_int().iloc[:, 0].values,
    "ci_upper_95": results.conf_int().iloc[:, 1].values,
})
coef_table["shock_type"] = np.select(
    [
        coef_table["term"].eq(MP_COL),
        coef_table["term"].eq(INFO_COL),
        coef_table["term"].str.startswith("MP_q"),
        coef_table["term"].str.startswith("INFO_q"),
        coef_table["term"].eq("asset_beta"),
        coef_table["term"].eq("BETA_MP"),
        coef_table["term"].eq("BETA_INFO"),
    ],
    ["MP_base", "INFO_base", "MP", "INFO", "beta", "beta_MP", "beta_INFO"],
    default="other"
)
coef_table["bin"] = coef_table["term"].str.extract(r"q(-?\d+)").astype(float)
coef_table["signif_10pct"] = coef_table["p_value"] < 0.10
coef_table["signif_5pct"] = coef_table["p_value"] < 0.05
coef_table["signif_1pct"] = coef_table["p_value"] < 0.01

sample_stats_rows = [
    {"metric": "n_firm_event_obs", "value": len(panel)},
    {"metric": "n_firms", "value": int(panel[GVKEY_COL].nunique())},
    {"metric": "n_events", "value": int(panel["event_date"].nunique())},
    {"metric": "dep_var_construction", "value": dep_construction},
    {"metric": "model_spec", "value": MODEL_SPEC},
    {"metric": "duration_transform", "value": USE_DURATION_TRANSFORM},
    {"metric": "include_firm_fe", "value": INCLUDE_FIRM_FE},
    {"metric": "include_event_fe", "value": INCLUDE_EVENT_FE},
    {"metric": "asset_beta_enabled", "value": ASSET_BETA},
]
if MODEL_SPEC in {"binned", "binned_withbeta"}:
    sample_stats_rows += [
        {"metric": "binning_mode", "value": BINNING_MODE},
        {"metric": "reference_bin", "value": reference_bin},
        {"metric": "n_present_bins", "value": len(present_bins)},
    ]
for k, v in series_stats(panel["dep_var"], "dep_var").items():
    sample_stats_rows.append({"metric": k, "value": v})

if MODEL_SPEC in {"binned", "binned_withbeta", "continuous"}:
    for k, v in series_stats(panel["duration_transformed"], "duration_transformed").items():
        sample_stats_rows.append({"metric": k, "value": v})

if ASSET_BETA and "asset_beta" in panel.columns:
    for k, v in series_stats(panel["asset_beta"], "asset_beta").items():
        sample_stats_rows.append({"metric": k, "value": v})

model_stats = pd.DataFrame([
    {"metric": "n_obs", "value": int(results.nobs)},
    {"metric": "df_model", "value": float(results.df_model)},
    {"metric": "df_resid", "value": float(results.df_resid)},
    {"metric": "rsquared_uncentered", "value": float(results.rsquared)},
    {"metric": "rsquared_adj_uncentered", "value": float(results.rsquared_adj)},
    {"metric": "cov_type", "value": results.cov_type},
    {"metric": "n_requested_regressors", "value": len(requested_x_cols)},
    {"metric": "n_estimated_regressors", "value": len(kept_x_cols)},
])

panel_event_stats = panel.groupby("event_date").agg(
    n_firms=(GVKEY_COL, "nunique"),
    dep_var_mean=("dep_var", "mean"),
    dep_var_std=("dep_var", "std"),
    shock_mp=(MP_COL, "first"),
    shock_info=(INFO_COL, "first"),
).reset_index().sort_values("event_date")

if MODEL_SPEC in {"binned", "binned_withbeta", "continuous"}:
    panel_event_stats["duration_mean"] = panel.groupby("event_date")["duration_transformed"].mean().values
    panel_event_stats["duration_std"] = panel.groupby("event_date")["duration_transformed"].std().values

if ASSET_BETA and "asset_beta" in panel.columns:
    panel_event_stats["asset_beta_mean"] = panel.groupby("event_date")["asset_beta"].mean().values
    panel_event_stats["asset_beta_std"] = panel.groupby("event_date")["asset_beta"].std().values

if MODEL_SPEC in {"binned", "binned_withbeta"}:
    # Dynamic aggregation names are passed as keyword arguments to pandas.
    agg_dict: dict[str, Any] = {
        "n_obs": pd.NamedAgg(GVKEY_COL, "size"),
        "n_firms": pd.NamedAgg(GVKEY_COL, "nunique"),
        "n_events": pd.NamedAgg("event_date", "nunique"),
        "dep_var_mean": pd.NamedAgg("dep_var", "mean"),
        "dep_var_std": pd.NamedAgg("dep_var", "std"),
        "duration_mean": pd.NamedAgg("duration_transformed", "mean"),
        "duration_median": pd.NamedAgg("duration_transformed", "median"),
        "duration_min": pd.NamedAgg("duration_transformed", "min"),
        "duration_max": pd.NamedAgg("duration_transformed", "max"),
        "mp_mean": pd.NamedAgg(MP_COL, "mean"),
        "info_mean": pd.NamedAgg(INFO_COL, "mean"),
    }
    if ASSET_BETA and "asset_beta" in panel.columns:
        agg_dict["asset_beta_mean"] = pd.NamedAgg("asset_beta", "mean")
        agg_dict["asset_beta_median"] = pd.NamedAgg("asset_beta", "median")
        agg_dict["asset_beta_min"] = pd.NamedAgg("asset_beta", "min")
        agg_dict["asset_beta_max"] = pd.NamedAgg("asset_beta", "max")

    panel_bin_stats = panel.groupby("duration_bin").agg(None, **agg_dict).reset_index().sort_values("duration_bin")
else:
    panel_bin_stats = pd.DataFrame(columns=[
        "duration_bin", "n_obs", "n_firms", "n_events", "dep_var_mean", "dep_var_std",
        "duration_mean", "duration_median", "duration_min", "duration_max", "mp_mean", "info_mean"
    ])

interaction_stats = []
for var in kept_x_cols:
    interaction_stats.append({
        "variable": var,
        "nonzero_count": int((panel[var] != 0).sum()),
        "mean": float(panel[var].mean()),
        "std": float(panel[var].std()) if panel[var].notna().sum() > 1 else np.nan,
        "min": float(panel[var].min()),
        "max": float(panel[var].max()),
        "residualized_nonzero": int((np.abs(resid_df[var]) > IDENTIFICATION_TOL).sum()),
    })
interaction_stats = pd.DataFrame(interaction_stats)

preview_cols = [
    GVKEY_COL, "event_date", "dur_date", "dep_var", MP_COL, INFO_COL,
    "duration_value_raw", "duration_for_model", "duration_transformed",
    "duration_bin", "days_since_duration_measure"
]
for extra_col in ["asset_beta", "BETA_MP", "BETA_INFO"]:
    if extra_col in panel.columns:
        preview_cols.append(extra_col)
preview_cols += kept_x_cols[:10]
panel_preview = panel[preview_cols].head(1000).copy()

# ============================================================
# STEP 10: SAVE OUTPUTS
# ============================================================
log_progress("STEP 10: Saving outputs to files.")
panel.to_csv(outpath(OUTPUT_REG_DATA), index=False)
coef_table.to_csv(outpath(OUTPUT_COEFS), index=False)
pd.DataFrame(input_audit_rows).to_csv(outpath(OUTPUT_INPUT_AUDIT_CSV), index=False)
pd.DataFrame(match_quality_rows).to_csv(outpath(OUTPUT_MATCH_QUALITY_CSV), index=False)
interaction_stats.to_csv(outpath(OUTPUT_INTERACTION_STATS_CSV), index=False)
pd.DataFrame(sample_stats_rows).to_csv(outpath(OUTPUT_SAMPLE_STATS_CSV), index=False)
model_stats.to_csv(outpath(OUTPUT_MODEL_STATS_CSV), index=False)
panel_event_stats.to_csv(outpath(OUTPUT_EVENT_STATS_CSV), index=False)
panel_bin_stats.to_csv(outpath(OUTPUT_BIN_STATS_CSV), index=False)
panel_preview.to_csv(outpath(OUTPUT_PANEL_PREVIEW_CSV), index=False)

pretty_stats_html = build_pretty_stats_html(
    panel_bin_stats=panel_bin_stats,
    sample_stats_rows=sample_stats_rows,
    model_stats=model_stats,
    model_spec=MODEL_SPEC,
    reference_bin=reference_bin,
    coef_table=coef_table,
)
with open(outpath(OUTPUT_STATS_TABLE_HTML), "w", encoding="utf-8") as f:
    f.write(pretty_stats_html)

summary_lines = []
summary_lines.append("Panel regression summary\n")
summary_lines.append("========================================\n")
summary_lines.append("Requested model:\n")
summary_lines.append(model_description(MODEL_SPEC, reference_bin=reference_bin) + "\n\n")
summary_lines.append(f"Dependent variable construction: {dep_construction}\n")
summary_lines.append(f"Duration transform: {USE_DURATION_TRANSFORM}\n")
summary_lines.append(f"Firm FE included: {INCLUDE_FIRM_FE}\n")
summary_lines.append(f"Event FE included: {INCLUDE_EVENT_FE}\n")
summary_lines.append(f"Asset beta enabled: {ASSET_BETA}\n")
if MODEL_SPEC in {"binned", "binned_withbeta"}:
    summary_lines.append(f"Binning mode: {BINNING_MODE}\n")
    summary_lines.append(f"Reference bin omitted: {reference_bin}\n")
    summary_lines.append(f"Occupied bins in sample: {len(present_bins)}\n")
summary_lines.append(f"Estimated regressors after FE absorption: {len(kept_x_cols)}\n\n")
summary_lines.append(f"Observations: {int(results.nobs):,}\n")
summary_lines.append(f"Firms: {panel[GVKEY_COL].nunique():,}\n")
summary_lines.append(f"Events: {panel['event_date'].nunique():,}\n")
summary_lines.append(f"R-squared (uncentered): {results.rsquared:.6f}\n")
summary_lines.append(f"Adj. R-squared (uncentered): {results.rsquared_adj:.6f}\n")
summary_lines.append(f"Covariance type: {results.cov_type}\n\n")
summary_lines.append("Coefficient table:\n")
summary_lines.append(coef_table.to_string(index=False))
summary_lines.append("\n\nFull statsmodels summary:\n")
summary_lines.append(str(results.summary()))

with open(outpath(OUTPUT_SUMMARY_TXT), "w", encoding="utf-8") as f:
    f.write("".join(summary_lines))
with open(outpath(OUTPUT_PROGRESS_LOG), "w", encoding="utf-8") as f:
    f.write("\n".join(PROGRESS_LOG) + "\n")

log_progress("All outputs saved successfully.")
print(f"\nDone. Outputs written to: {OUTDIR.resolve()}")