import numpy as np
import pandas as pd

# =========================
# SETTINGS
# =========================
INPUT_FILE = "Implied_Duration_g=0.4.csv"   # change to your file path
OUTPUT_SUMMARY = "duration_diagnostic_summary.txt"
OUTPUT_FLAGGED = "duration_flagged_rows.csv"

# Thresholds
EXTREME_DURATION_HIGH = 50
EXTREME_DURATION_LOW = 0          # duration <= 0 flagged
LARGE_MONTHLY_JUMP = 5.0          # absolute monthly change in implied duration
STALE_QUARTERLY_MONTHS = 4        # if quarterly_datadate is older than this many months relative to month_date
RATIO_TOL = 1e-6
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99


# =========================
# LOAD
# =========================
df = pd.read_csv(INPUT_FILE)

# Expected columns
expected_cols = [
    "gvkey", "month_date", "implied_duration_years", "conm", "quarterly_datadate",
    "fyearq", "fqtr", "market_cap", "sales", "sale_lag", "earnings",
    "book_equity", "book_equity_lag", "sales_growth_0", "roe0",
    "book_to_market", "earnings_to_price", "status", "message"
]

missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

# Parse dates
df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
df["quarterly_datadate"] = pd.to_datetime(df["quarterly_datadate"], errors="coerce")

# Convert numeric columns
numeric_cols = [
    "gvkey", "implied_duration_years", "fyearq", "fqtr", "market_cap", "sales",
    "sale_lag", "earnings", "book_equity", "book_equity_lag", "sales_growth_0",
    "roe0", "book_to_market", "earnings_to_price"
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Sort
df = df.sort_values(["gvkey", "month_date"]).reset_index(drop=True)


# =========================
# BASIC DIAGNOSTICS
# =========================
report_lines = []
flag_masks = {}

report_lines.append("========== IMPLIED DURATION FILE DIAGNOSTICS ==========\n")
report_lines.append(f"Rows: {len(df):,}")
report_lines.append(f"Unique gvkeys: {df['gvkey'].nunique():,}")
report_lines.append(f"Date range (month_date): {df['month_date'].min()} to {df['month_date'].max()}")
report_lines.append("")

# Missing values
missing_summary = df[expected_cols].isna().sum().sort_values(ascending=False)
report_lines.append("---- Missing values by column ----")
report_lines.extend([f"{col}: {cnt:,}" for col, cnt in missing_summary.items()])
report_lines.append("")

# Duplicate keys
dup_key = df.duplicated(subset=["gvkey", "month_date"], keep=False)
flag_masks["duplicate_gvkey_month"] = dup_key

report_lines.append("---- Duplicate (gvkey, month_date) rows ----")
report_lines.append(f"Count: {dup_key.sum():,}")
report_lines.append("")

# Bad statuses
bad_status = ~df["status"].fillna("").str.lower().eq("ok")
flag_masks["bad_status"] = bad_status

report_lines.append("---- Non-ok status rows ----")
report_lines.append(f"Count: {bad_status.sum():,}")
if bad_status.any():
    report_lines.append("Status breakdown:")
    report_lines.extend(df.loc[bad_status, "status"].value_counts(dropna=False).astype(str).to_dict().__repr__().splitlines())
report_lines.append("")

# Date parse issues
bad_month_date = df["month_date"].isna()
bad_quarterly_date = df["quarterly_datadate"].isna()
flag_masks["bad_month_date"] = bad_month_date
flag_masks["bad_quarterly_datadate"] = bad_quarterly_date

report_lines.append("---- Date parse issues ----")
report_lines.append(f"Bad month_date: {bad_month_date.sum():,}")
report_lines.append(f"Bad quarterly_datadate: {bad_quarterly_date.sum():,}")
report_lines.append("")


# =========================
# NUMERIC SANITY CHECKS
# =========================
nonpos_market_cap = df["market_cap"] <= 0
nonpos_book_equity_lag = df["book_equity_lag"] <= 0
nonpos_market_cap_or_na = df["market_cap"].isna() | (df["market_cap"] <= 0)
nonpos_book_equity_lag_or_na = df["book_equity_lag"].isna() | (df["book_equity_lag"] <= 0)
sale_lag_zero = df["sale_lag"].isna() | (df["sale_lag"] == 0)

flag_masks["nonpositive_market_cap"] = nonpos_market_cap
flag_masks["nonpositive_book_equity_lag"] = nonpos_book_equity_lag
flag_masks["sale_lag_zero_or_na"] = sale_lag_zero

report_lines.append("---- Nonpositive denominator checks ----")
report_lines.append(f"market_cap <= 0: {nonpos_market_cap.sum():,}")
report_lines.append(f"book_equity_lag <= 0: {nonpos_book_equity_lag.sum():,}")
report_lines.append(f"sale_lag == 0 or NA: {sale_lag_zero.sum():,}")
report_lines.append("")

# Recompute ratios
calc_sales_growth = (df["sales"] - df["sale_lag"]) / df["sale_lag"]
calc_roe0 = df["earnings"] / df["book_equity_lag"]
calc_btm = df["book_equity"] / df["market_cap"]
calc_etp = df["earnings"] / df["market_cap"]

sales_growth_mismatch = ~(np.isclose(df["sales_growth_0"], calc_sales_growth, atol=RATIO_TOL, equal_nan=True))
roe0_mismatch = ~(np.isclose(df["roe0"], calc_roe0, atol=RATIO_TOL, equal_nan=True))
btm_mismatch = ~(np.isclose(df["book_to_market"], calc_btm, atol=RATIO_TOL, equal_nan=True))
etp_mismatch = ~(np.isclose(df["earnings_to_price"], calc_etp, atol=RATIO_TOL, equal_nan=True))

# avoid flagging rows where denominator itself invalid
sales_growth_mismatch = sales_growth_mismatch & ~sale_lag_zero
roe0_mismatch = roe0_mismatch & ~nonpos_book_equity_lag_or_na
btm_mismatch = btm_mismatch & ~nonpos_market_cap_or_na
etp_mismatch = etp_mismatch & ~nonpos_market_cap_or_na

flag_masks["sales_growth_mismatch"] = sales_growth_mismatch
flag_masks["roe0_mismatch"] = roe0_mismatch
flag_masks["book_to_market_mismatch"] = btm_mismatch
flag_masks["earnings_to_price_mismatch"] = etp_mismatch

report_lines.append("---- Ratio consistency checks ----")
report_lines.append(f"sales_growth_0 mismatch: {sales_growth_mismatch.sum():,}")
report_lines.append(f"roe0 mismatch: {roe0_mismatch.sum():,}")
report_lines.append(f"book_to_market mismatch: {btm_mismatch.sum():,}")
report_lines.append(f"earnings_to_price mismatch: {etp_mismatch.sum():,}")
report_lines.append("")


# =========================
# IMPLIED DURATION CHECKS
# =========================
neg_or_zero_duration = df["implied_duration_years"] <= EXTREME_DURATION_LOW
extreme_high_duration = df["implied_duration_years"] > EXTREME_DURATION_HIGH
missing_duration = df["implied_duration_years"].isna()

flag_masks["duration_nonpositive"] = neg_or_zero_duration
flag_masks["duration_extreme_high"] = extreme_high_duration
flag_masks["duration_missing"] = missing_duration

report_lines.append("---- Implied duration checks ----")
report_lines.append(f"Missing duration: {missing_duration.sum():,}")
report_lines.append(f"Duration <= {EXTREME_DURATION_LOW}: {neg_or_zero_duration.sum():,}")
report_lines.append(f"Duration > {EXTREME_DURATION_HIGH}: {extreme_high_duration.sum():,}")
report_lines.append("")

dur_desc = df["implied_duration_years"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
report_lines.append("Duration distribution:")
report_lines.extend([f"{idx}: {val}" for idx, val in dur_desc.items()])
report_lines.append("")

# Winsorization view only, not changing data
q_low = df["implied_duration_years"].quantile(WINSOR_Q_LOW)
q_high = df["implied_duration_years"].quantile(WINSOR_Q_HIGH)
report_lines.append(f"Winsor bounds ({WINSOR_Q_LOW:.0%}, {WINSOR_Q_HIGH:.0%}): {q_low:.6f}, {q_high:.6f}")
report_lines.append("")


# =========================
# TIME-SERIES CONSISTENCY WITHIN GVKEY
# =========================
df["duration_lag"] = df.groupby("gvkey")["implied_duration_years"].shift(1)
df["duration_change"] = df["implied_duration_years"] - df["duration_lag"]
large_jump = df["duration_change"].abs() > LARGE_MONTHLY_JUMP
flag_masks["large_monthly_duration_jump"] = large_jump

report_lines.append("---- Within-gvkey duration jump checks ----")
report_lines.append(f"|Monthly duration change| > {LARGE_MONTHLY_JUMP}: {large_jump.sum():,}")
report_lines.append("")

# Check month_date monotonicity within gvkey
date_diff = df.groupby("gvkey")["month_date"].diff()
nonmonotonic = date_diff.dt.days < 0
flag_masks["nonmonotonic_month_date"] = nonmonotonic.fillna(False)

report_lines.append("---- Within-gvkey date order ----")
report_lines.append(f"Nonmonotonic month_date rows: {nonmonotonic.fillna(False).sum():,}")
report_lines.append("")


# =========================
# QUARTERLY DATA STALENESS / ALIGNMENT
# =========================
# quarterly_datadate should usually be <= month_date
future_quarterly = df["quarterly_datadate"] > df["month_date"]
flag_masks["future_quarterly_datadate"] = future_quarterly

# month gap approximation
month_gap = (
    (df["month_date"].dt.year - df["quarterly_datadate"].dt.year) * 12
    + (df["month_date"].dt.month - df["quarterly_datadate"].dt.month)
)
df["quarterly_age_months"] = month_gap

stale_quarterly = month_gap > STALE_QUARTERLY_MONTHS
flag_masks["stale_quarterly_mapping"] = stale_quarterly

report_lines.append("---- Quarterly mapping checks ----")
report_lines.append(f"quarterly_datadate > month_date: {future_quarterly.sum():,}")
report_lines.append(f"quarterly data older than {STALE_QUARTERLY_MONTHS} months: {stale_quarterly.sum():,}")
report_lines.append("")

quarterly_age_desc = df["quarterly_age_months"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
report_lines.append("Quarterly age in months distribution:")
report_lines.extend([f"{idx}: {val}" for idx, val in quarterly_age_desc.items()])
report_lines.append("")


# =========================
# ECONOMICALLY SUSPICIOUS INPUTS
# =========================
very_large_sales_growth = df["sales_growth_0"].abs() > 2
very_large_roe = df["roe0"].abs() > 1
negative_book_equity = df["book_equity"] <= 0
negative_book_equity_lag = df["book_equity_lag"] <= 0
negative_market_cap = df["market_cap"] <= 0

flag_masks["very_large_sales_growth"] = very_large_sales_growth
flag_masks["very_large_roe"] = very_large_roe
flag_masks["negative_book_equity"] = negative_book_equity
flag_masks["negative_book_equity_lag"] = negative_book_equity_lag
flag_masks["negative_market_cap"] = negative_market_cap

report_lines.append("---- Suspicious accounting / ratio inputs ----")
report_lines.append(f"|sales_growth_0| > 2: {very_large_sales_growth.sum():,}")
report_lines.append(f"|roe0| > 1: {very_large_roe.sum():,}")
report_lines.append(f"book_equity <= 0: {negative_book_equity.sum():,}")
report_lines.append(f"book_equity_lag <= 0: {negative_book_equity_lag.sum():,}")
report_lines.append(f"market_cap <= 0: {negative_market_cap.sum():,}")
report_lines.append("")


# =========================
# COMBINE FLAGS
# =========================
flag_df = df.copy()

flag_columns = []
for name, mask in flag_masks.items():
    col = f"flag_{name}"
    flag_df[col] = mask.fillna(False)
    flag_columns.append(col)

flag_df["n_flags"] = flag_df[flag_columns].sum(axis=1)

# useful summary label
def collect_issues(row):
    issues = [name.replace("flag_", "") for name in flag_columns if row[name]]
    return "; ".join(issues)

flag_df["issues"] = flag_df.apply(collect_issues, axis=1)

report_lines.append("---- Overall flag summary ----")
report_lines.append(f"Rows with at least 1 flag: {(flag_df['n_flags'] > 0).sum():,}")
report_lines.append(f"Rows with at least 2 flags: {(flag_df['n_flags'] > 1).sum():,}")
report_lines.append(f"Rows with at least 3 flags: {(flag_df['n_flags'] > 2).sum():,}")
report_lines.append("")

top_issue_counts = (
    pd.Series({k: v.fillna(False).sum() for k, v in flag_masks.items()})
    .sort_values(ascending=False)
)

report_lines.append("Top issue counts:")
report_lines.extend([f"{idx}: {val:,}" for idx, val in top_issue_counts.items()])
report_lines.append("")


# =========================
# OPTIONAL: FOCUS ON DURATION PROBLEM ROWS
# =========================
duration_problem_mask = (
    flag_df["flag_duration_nonpositive"]
    | flag_df["flag_duration_extreme_high"]
    | flag_df["flag_large_monthly_duration_jump"]
    | flag_df["flag_stale_quarterly_mapping"]
    | flag_df["flag_bad_status"]
)

duration_problem_rows = flag_df.loc[duration_problem_mask].copy()

report_lines.append("---- Focused duration-problem sample size ----")
report_lines.append(f"Rows flagged by duration-related filters: {len(duration_problem_rows):,}")
report_lines.append("")


# =========================
# EXPORT
# =========================
with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

flag_df.to_csv(OUTPUT_FLAGGED, index=False)

print(f"Saved summary to: {OUTPUT_SUMMARY}")
print(f"Saved flagged rows to: {OUTPUT_FLAGGED}")

# Print a short console summary too
print("\n========== SHORT SUMMARY ==========")
print(f"Rows: {len(df):,}")
print(f"Unique gvkeys: {df['gvkey'].nunique():,}")
print(f"Rows with >=1 flag: {(flag_df['n_flags'] > 0).sum():,}")
print("\nTop issues:")
print(top_issue_counts.to_string())

print("\nSample flagged rows:")
show_cols = [
    "gvkey", "conm", "month_date", "quarterly_datadate",
    "implied_duration_years", "duration_change", "quarterly_age_months",
    "sales_growth_0", "roe0", "book_to_market", "earnings_to_price",
    "status", "message", "n_flags", "issues"
]
print(flag_df.loc[flag_df["n_flags"] > 0, show_cols].head(20).to_string(index=False))
