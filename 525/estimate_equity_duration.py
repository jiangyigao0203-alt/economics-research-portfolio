"""Estimate implied equity duration from a monthly accounting/security panel.

Project mean-reverting sales growth and return on equity, discount the implied
cash flows, and combine finite-horizon and residual terminal-value duration.
DEFAULT_PARAMS controls the economic assumptions. Optional June NYSE microcap
classification uses an external breakpoint file; outlier rules are configurable.
Inputs and outputs are relative to the 525 working directory. Results include
duration estimates and diagnostics; the underlying firm-level data stay local.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================
INPUT_PANEL_FILE = "compustat_monthly_panel_from_quarterly_2000_2024.csv"
INPUT_NYSE_BREAKPOINT_FILE = "ME_Breakpoints.csv"

OUTPUT_FILE = "monthly_implied_duration_years.csv"
OUTPUT_MICROCAP_DIAGNOSTICS = "monthly_implied_duration_microcap_diagnostics.csv"

DEFAULT_PARAMS = {
    "rho_roe": 0.39,
    "rho_g": 0.21,
    "r": 0.12,
    "g_bar": 0.06,
    "T": 15,
}

OUTLIER_METHOD = "drop"   # "drop", "winsorize", or None

# ============================================================
# EXTERNAL NYSE MICROCAP BREAKPOINT FILE SETTINGS
# ============================================================
# The file should contain at least:
#   date  -> YYYYMM
#   20th  -> NYSE 20th percentile market equity
#
# Example:
#   date,20th
#   192512,4.96
#   192601,4.84
#   ...
#
# We use JUNE rows only for the annual classification:
# - June of year t defines microcaps
# - apply to July t through June t+1

APPLY_JUNE_NYSE_MICROCAP_EXCLUSION = False
BREAKPOINT_DATE_COL = "date"
BREAKPOINT_20TH_COL = "20th"

# If breakpoint values use a different scale from market_cap,
# multiply the breakpoint by this factor before comparing to market_cap.
# Usually leave at 1.0 if both are in millions.
BREAKPOINT_SCALE_MULTIPLIER = 1.0


# ============================================================
# HELPERS
# ============================================================
def normalize_gvkey(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.astype("Int64").astype(str).str.zfill(6)


def _safe_divide(a, b):
    if b is None or pd.isna(b) or abs(b) < 1e-12:
        return np.nan
    return a / b


def _apply_duration_outlier_rule(df, method="drop", lower_q=0.01, upper_q=0.99):
    """Drop or winsorize nonmissing duration values using the specified sample quantiles."""
    if method is None:
        print("No duration outlier handling applied.")
        return df.copy(), np.nan, np.nan, 0

    out = df.copy()
    valid = out["implied_duration_years"].notna()

    if valid.sum() == 0:
        print("No valid implied_duration_years values found for outlier handling.")
        return out, np.nan, np.nan, 0

    lower = out.loc[valid, "implied_duration_years"].quantile(lower_q)
    upper = out.loc[valid, "implied_duration_years"].quantile(upper_q)

    if method == "drop":
        before = len(out)
        out = out[
            out["implied_duration_years"].isna()
            | (
                (out["implied_duration_years"] >= lower)
                & (out["implied_duration_years"] <= upper)
            )
        ].copy()
        removed = before - len(out)
        print(
            f"Dropped duration outliers outside [{lower:.6f}, {upper:.6f}] "
            f"at the {lower_q:.0%}/{upper_q:.0%} cutoffs."
        )
        return out, lower, upper, removed

    if method == "winsorize":
        changed = (
            out["implied_duration_years"].notna()
            & (
                (out["implied_duration_years"] < lower)
                | (out["implied_duration_years"] > upper)
            )
        ).sum()
        out.loc[valid, "implied_duration_years"] = out.loc[valid, "implied_duration_years"].clip(
            lower=lower, upper=upper
        )
        print(
            f"Winsorized implied_duration_years to [{lower:.6f}, {upper:.6f}] "
            f"at the {lower_q:.0%}/{upper_q:.0%} cutoffs."
        )
        return out, lower, upper, int(changed)

    raise ValueError("OUTLIER_METHOD must be one of: 'drop', 'winsorize', None")


def _holding_year_from_month_date(month_series):
    # July t to June t+1 belongs to holding_year = t
    m = month_series.dt.month
    y = month_series.dt.year
    return np.where(m >= 7, y, y - 1)


def _load_external_breakpoint_file(path):
    print(f"\nReading external NYSE breakpoint file: {path}")
    bp = pd.read_csv(path)

    if BREAKPOINT_DATE_COL not in bp.columns:
        raise KeyError(f"Missing breakpoint date column: {BREAKPOINT_DATE_COL}")
    if BREAKPOINT_20TH_COL not in bp.columns:
        raise KeyError(f"Missing breakpoint 20th column: {BREAKPOINT_20TH_COL}")

    bp = bp[[BREAKPOINT_DATE_COL, BREAKPOINT_20TH_COL]].copy()
    bp[BREAKPOINT_DATE_COL] = pd.to_numeric(bp[BREAKPOINT_DATE_COL], errors="coerce").astype("Int64")
    bp[BREAKPOINT_20TH_COL] = pd.to_numeric(bp[BREAKPOINT_20TH_COL], errors="coerce")

    bp = bp.dropna(subset=[BREAKPOINT_DATE_COL, BREAKPOINT_20TH_COL]).copy()

    # Parse YYYYMM
    bp["date_str"] = bp[BREAKPOINT_DATE_COL].astype(str).str.zfill(6)
    bp["year"] = pd.to_numeric(bp["date_str"].str[:4], errors="coerce")
    bp["month"] = pd.to_numeric(bp["date_str"].str[4:6], errors="coerce")

    # Keep June only
    bp = bp[bp["month"] == 6].copy()
    if len(bp) == 0:
        raise ValueError("No June rows found in breakpoint file.")

    bp["june_year"] = bp["year"].astype(int)
    bp["nyse_microcap_breakpoint"] = bp[BREAKPOINT_20TH_COL] * BREAKPOINT_SCALE_MULTIPLIER
    bp["breakpoint_source"] = "EXTERNAL_FF_FILE"

    # Keep one row per June year
    bp = (
        bp.sort_values("june_year")
        .drop_duplicates(subset=["june_year"], keep="last")
        [["june_year", "nyse_microcap_breakpoint", "breakpoint_source"]]
        .reset_index(drop=True)
    )

    print(f"  June breakpoint years loaded: {len(bp):,}")
    print(f"  Breakpoint year range: {bp['june_year'].min()} to {bp['june_year'].max()}")
    return bp


def _build_june_classification_from_external_breakpoints(panel, breakpoint_df):
    """
    For each firm's June observation:
    - get June market cap
    - merge external June breakpoint
    - classify as microcap if June market cap < June breakpoint
    - this classification applies to holding year = June year
    """
    work = panel.copy()

    if "month_date" not in work.columns:
        raise KeyError("month_date is required.")
    if "market_cap" not in work.columns:
        raise KeyError("market_cap is required.")

    work["month_date"] = pd.to_datetime(work["month_date"], errors="coerce")
    work["market_cap"] = pd.to_numeric(work["market_cap"], errors="coerce")

    work = work.dropna(subset=["gvkey", "month_date", "market_cap"]).copy()
    work = work[work["market_cap"] > 0].copy()

    june = work[work["month_date"].dt.month == 6].copy()
    june["june_year"] = june["month_date"].dt.year

    # One firm per June year
    june = (
        june.sort_values(["gvkey", "month_date"])
        .drop_duplicates(subset=["gvkey", "june_year"], keep="last")
        .copy()
    )

    if len(june) == 0:
        raise ValueError("No firm June observations found in prepared panel.")

    june = june.merge(breakpoint_df, on="june_year", how="left")

    june["is_microcap_june"] = june["market_cap"] < june["nyse_microcap_breakpoint"]
    june["holding_year"] = june["june_year"]

    june_class = june[
        [
            "gvkey",
            "june_year",
            "holding_year",
            "month_date",
            "market_cap",
            "nyse_microcap_breakpoint",
            "is_microcap_june",
            "breakpoint_source",
        ]
    ].rename(
        columns={
            "month_date": "june_date_used",
            "market_cap": "june_market_cap",
        }
    )

    return june_class


def _apply_june_microcap_exclusion(monthly_results, prepared_panel, breakpoint_file):
    """
    Apply June-based microcap exclusion using external FF breakpoint file.

    Logic:
    - read external June NYSE 20th percentile file
    - classify each firm's June observation as microcap or not
    - assign that classification to July t ... June t+1
    - drop firm-months classified as microcap
    """
    print("\nApplying June NYSE microcap exclusion using external breakpoint file...")

    breakpoint_df = _load_external_breakpoint_file(breakpoint_file)
    june_class = _build_june_classification_from_external_breakpoints(prepared_panel, breakpoint_df)

    out = monthly_results.copy()
    out["month_date"] = pd.to_datetime(out["month_date"], errors="coerce")
    out["holding_year"] = _holding_year_from_month_date(out["month_date"])

    out = out.merge(
        june_class[
            [
                "gvkey",
                "holding_year",
                "june_year",
                "june_date_used",
                "june_market_cap",
                "nyse_microcap_breakpoint",
                "is_microcap_june",
                "breakpoint_source",
            ]
        ],
        on=["gvkey", "holding_year"],
        how="left",
    )

    # If no June classification found, keep row by default
    out["is_microcap_june"] = out["is_microcap_june"].fillna(False)

    before = len(out)
    excluded = int(out["is_microcap_june"].sum())
    out = out.loc[~out["is_microcap_june"]].copy()
    after = len(out)

    print(f"  Rows before microcap exclusion: {before:,}")
    print(f"  Rows excluded as microcaps:     {excluded:,}")
    print(f"  Rows after microcap exclusion:  {after:,}")

    diag_rows = [
        {"metric": "microcap_filter_applied", "value": True},
        {"metric": "breakpoint_file", "value": breakpoint_file},
        {"metric": "breakpoint_date_col", "value": BREAKPOINT_DATE_COL},
        {"metric": "breakpoint_20th_col", "value": BREAKPOINT_20TH_COL},
        {"metric": "breakpoint_scale_multiplier", "value": BREAKPOINT_SCALE_MULTIPLIER},
        {"metric": "rows_before_microcap_exclusion", "value": before},
        {"metric": "rows_excluded_microcap", "value": excluded},
        {"metric": "rows_after_microcap_exclusion", "value": after},
        {"metric": "june_breakpoint_years_loaded", "value": int(breakpoint_df["june_year"].nunique())},
        {"metric": "june_classified_firm_years", "value": len(june_class)},
        {"metric": "kept_rows_with_june_classification", "value": int(out["june_year"].notna().sum())},
    ]

    return out, pd.DataFrame(diag_rows)


# ============================================================
# LOAD / PREP PANEL
# ============================================================
def load_panel(path):
    print(f"Reading panel file: {path}")
    df = pd.read_csv(path)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns found:\n{list(df.columns)}\n")

    if "gvkey" not in df.columns:
        raise KeyError("Missing gvkey in input panel.")
    df["gvkey"] = normalize_gvkey(df["gvkey"])

    date_cols = ["quarterly_datadate", "month_date", "security_datadate"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    print("Date parsing check:")
    for c in date_cols:
        if c in df.columns:
            n_bad = df[c].isna().sum()
            print(f"  {c}: bad/missing parsed dates = {n_bad:,}")

    df = df.sort_values(["gvkey", "month_date"]).reset_index(drop=True)
    return df


def prepare_panel(df):
    """Prepare firm-month accounting inputs and lagged quantities for duration estimation."""
    out = df.copy()

    required_cols = [
        "gvkey", "conm", "quarterly_datadate", "fyearq", "fqtr", "month_date",
        "saleq", "ibq", "seqq", "prccq_quarterly", "cshoq_quarterly",
        "prcc_m", "csho_m", "market_cap"
    ]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    print("\nChecking share / market cap fields...")
    if "csho_m_raw" in out.columns:
        print("  Found csho_m_raw (raw shares).")
    print("  Found csho_m (already in millions).")
    print("  Found market_cap.")

    out["csho_m_mil"] = out["csho_m"]

    out["market_cap_recomputed"] = out["prcc_m"].abs() * out["csho_m_mil"]
    out["market_cap_q_fallback"] = out["prccq_quarterly"].abs() * out["cshoq_quarterly"]

    out["market_cap_final"] = out["market_cap"]
    out["market_cap_final"] = out["market_cap_final"].combine_first(out["market_cap_recomputed"])
    out["market_cap_final"] = out["market_cap_final"].combine_first(out["market_cap_q_fallback"])

    diff = (out["market_cap"] - out["market_cap_recomputed"]).abs()
    valid_cmp = out["market_cap"].notna() & out["market_cap_recomputed"].notna()
    if valid_cmp.sum() > 0:
        print(f"  Mean |market_cap - prcc_m*csho_m| = {diff[valid_cmp].mean():.6f}")
        print(f"  Max  |market_cap - prcc_m*csho_m| = {diff[valid_cmp].max():.6f}")

    state_cols = [
        "gvkey", "quarterly_datadate", "fyearq", "fqtr", "saleq", "ibq", "seqq"
    ]
    state = (
        out[state_cols]
        .drop_duplicates(subset=["gvkey", "quarterly_datadate", "fyearq", "fqtr"])
        .sort_values(["gvkey", "quarterly_datadate"])
        .reset_index(drop=True)
    )

    state["sale_lag"] = state.groupby("gvkey")["saleq"].shift(1)
    state["earnings_lag"] = state.groupby("gvkey")["ibq"].shift(1)
    state["book_equity_lag"] = state.groupby("gvkey")["seqq"].shift(1)

    out = out.merge(
        state[
            [
                "gvkey", "quarterly_datadate", "fyearq", "fqtr",
                "sale_lag", "earnings_lag", "book_equity_lag"
            ]
        ],
        on=["gvkey", "quarterly_datadate", "fyearq", "fqtr"],
        how="left",
    )

    out = out.rename(
        columns={
            "saleq": "sales",
            "ibq": "earnings",
            "seqq": "book_equity",
        }
    )

    out["market_cap"] = out["market_cap_final"]

    print("\nPrepared panel summary:")
    print(f"  Rows: {len(out):,}")
    print(f"  Unique gvkey: {out['gvkey'].nunique():,}")
    print(f"  Non-missing market_cap: {out['market_cap'].notna().sum():,}")
    print(f"  Non-missing sale_lag: {out['sale_lag'].notna().sum():,}")
    print(f"  Non-missing book_equity_lag: {out['book_equity_lag'].notna().sum():,}")

    return out


# ============================================================
# CORE DURATION MODEL
# ============================================================
def _validate_inputs(P0, B_minus_1, B0, E0, S_minus_1, S0):
    if pd.isna(P0) or P0 <= 0:
        return False, "Invalid P0: market cap must be positive."
    if pd.isna(B_minus_1) or B_minus_1 <= 0:
        return False, "Invalid B_minus_1: lagged book equity must be positive."
    if pd.isna(B0) or B0 <= 0:
        return False, "Invalid B0: current book equity must be positive."
    if pd.isna(E0):
        return False, "Invalid E0: earnings missing."
    if pd.isna(S_minus_1) or S_minus_1 <= 0:
        return False, "Invalid S_minus_1: lagged sales must be positive."
    if pd.isna(S0) or S0 <= 0:
        return False, "Invalid S0: current sales must be positive."
    return True, "OK"


def compute_implied_duration_row(
    P0,
    B_minus_1,
    B0,
    E0,
    S_minus_1,
    S0,
    rho_roe=DEFAULT_PARAMS["rho_roe"],
    rho_g=DEFAULT_PARAMS["rho_g"],
    r=DEFAULT_PARAMS["r"],
    g_bar=DEFAULT_PARAMS["g_bar"],
    T=DEFAULT_PARAMS["T"],
):
    """Return duration components for one firm observation under the configured cash-flow assumptions.

    P0 is market value; B values are book equity, E0 is earnings, and S values
    are sales. Invalid economic inputs return an error status and missing duration."""
    valid, msg = _validate_inputs(P0, B_minus_1, B0, E0, S_minus_1, S0)
    if not valid:
        return {
            "status": "error",
            "message": msg,
            "implied_duration_years": np.nan,
        }

    sales_growth_0 = (S0 - S_minus_1) / S_minus_1
    roe0 = E0 / B_minus_1

    if pd.isna(sales_growth_0) or pd.isna(roe0):
        return {
            "status": "error",
            "message": "Initial growth/ROE missing.",
            "implied_duration_years": np.nan,
        }

    growth = np.full(T + 1, np.nan)
    roe = np.full(T + 1, np.nan)
    bv = np.full(T + 1, np.nan)
    cf = np.full(T + 1, np.nan)
    pv_cf = np.full(T + 1, np.nan)
    t_pv_cf = np.full(T + 1, np.nan)

    growth[0] = sales_growth_0
    roe[0] = roe0
    bv[0] = B0

    for t in range(1, T + 1):
        growth[t] = g_bar + rho_g * (growth[t - 1] - g_bar)
        roe[t] = r + rho_roe * (roe[t - 1] - r)

    prev_bv = B0
    for t in range(1, T + 1):
        bv[t] = prev_bv * (1 + growth[t])
        cf[t] = prev_bv * (roe[t] - growth[t])
        pv_cf[t] = cf[t] / ((1 + r) ** t)
        t_pv_cf[t] = t * pv_cf[t]
        prev_bv = bv[t]

    pv_sum = np.nansum(pv_cf[1:])
    t_pv_sum = np.nansum(t_pv_cf[1:])
    finite_duration = np.nan if abs(pv_sum) < 1e-12 else t_pv_sum / pv_sum

    terminal_pv = P0 - pv_sum
    terminal_duration = T + (1 + r) / r

    finite_weight = pv_sum / P0
    terminal_weight = terminal_pv / P0

    implied_duration = (
        finite_weight * finite_duration + terminal_weight * terminal_duration
        if not pd.isna(finite_duration)
        else np.nan
    )

    earnings_to_price = _safe_divide(E0, P0)
    book_to_market = _safe_divide(B0, P0)

    ep_approx = (
        T + (1 + r) / r - T * earnings_to_price / r
        if not pd.isna(earnings_to_price)
        else np.nan
    )

    bm_approx = (
        T + (1 + r) / r - T * book_to_market
        if not pd.isna(book_to_market)
        else np.nan
    )

    return {
        "status": "ok",
        "message": "success",
        "sales_growth_0": sales_growth_0,
        "roe0": roe0,
        "pv_sum_1_to_T": pv_sum,
        "t_pv_sum_1_to_T": t_pv_sum,
        "finite_duration": finite_duration,
        "terminal_pv": terminal_pv,
        "terminal_duration": terminal_duration,
        "finite_weight": finite_weight,
        "terminal_weight": terminal_weight,
        "implied_duration_years": implied_duration,
        "earnings_to_price": earnings_to_price,
        "book_to_market": book_to_market,
        "ep_approx_duration": ep_approx,
        "bm_approx_duration": bm_approx,
    }


def compute_duration_for_panel(df, params=None):
    """Apply the row-level duration model to a prepared firm-month panel."""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    results = []
    print("\nComputing monthly implied duration series...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Duration progress"):
        res = compute_implied_duration_row(
            P0=row["market_cap"],
            B_minus_1=row["book_equity_lag"],
            B0=row["book_equity"],
            E0=row["earnings"],
            S_minus_1=row["sale_lag"],
            S0=row["sales"],
            rho_roe=params["rho_roe"],
            rho_g=params["rho_g"],
            r=params["r"],
            g_bar=params["g_bar"],
            T=params["T"],
        )

        results.append({
            "row_index": idx,
            "gvkey": row.get("gvkey"),
            "conm": row.get("conm"),
            "month_date": row.get("month_date"),
            "quarterly_datadate": row.get("quarterly_datadate"),
            "fyearq": row.get("fyearq"),
            "fqtr": row.get("fqtr"),
            "sales": row.get("sales"),
            "sale_lag": row.get("sale_lag"),
            "earnings": row.get("earnings"),
            "earnings_lag": row.get("earnings_lag"),
            "book_equity": row.get("book_equity"),
            "book_equity_lag": row.get("book_equity_lag"),
            "prcc_m": row.get("prcc_m"),
            "csho_m_raw": row.get("csho_m_raw"),
            "csho_m_mil": row.get("csho_m_mil"),
            "market_cap": row.get("market_cap"),
            **res
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================
def main():
    raw = load_panel(INPUT_PANEL_FILE)
    panel = prepare_panel(raw)

    results_all = compute_duration_for_panel(panel, DEFAULT_PARAMS)

    total_obs = len(results_all)
    valid_mask = results_all["status"] == "ok"
    dropped_invalid_obs = total_obs - valid_mask.sum()
    monthly_results = results_all.loc[valid_mask].copy()

    print(f"\nValid duration rows before outlier handling: {len(monthly_results):,} / {total_obs:,}")

    monthly_results, lower, upper, n_changed = _apply_duration_outlier_rule(
        monthly_results,
        method=OUTLIER_METHOD,
        lower_q=0.01,
        upper_q=0.99,
    )

    monthly_results["month_date"] = pd.to_datetime(monthly_results["month_date"], errors="coerce").dt.normalize()
    monthly_results = monthly_results.sort_values(["gvkey", "month_date"]).reset_index(drop=True)

    # Drop duplicate firm-month rows if any
    before = len(monthly_results)
    monthly_results = monthly_results.drop_duplicates(subset=["gvkey", "month_date"], keep="last").copy()
    dropped_dups = before - len(monthly_results)

    # ------------------------------------------------------------
    # APPLY EXTERNAL JUNE-BASED NYSE MICROCAP EXCLUSION
    # ------------------------------------------------------------
    microcap_diag = pd.DataFrame(
        [{"metric": "microcap_filter_applied", "value": False}]
    )

    if APPLY_JUNE_NYSE_MICROCAP_EXCLUSION:
        monthly_results, microcap_diag = _apply_june_microcap_exclusion(
            monthly_results=monthly_results,
            prepared_panel=panel,
            breakpoint_file=INPUT_NYSE_BREAKPOINT_FILE,
        )

    output_cols = [
        "gvkey",
        "month_date",
        "implied_duration_years",
        "conm",
        "quarterly_datadate",
        "fyearq",
        "fqtr",
        "market_cap",
        "sales",
        "sale_lag",
        "earnings",
        "book_equity",
        "book_equity_lag",
        "sales_growth_0",
        "roe0",
        "book_to_market",
        "earnings_to_price",
        "holding_year",
        "june_year",
        "june_date_used",
        "june_market_cap",
        "nyse_microcap_breakpoint",
        "breakpoint_source",
        "status",
        "message",
    ]
    output_cols = [c for c in output_cols if c in monthly_results.columns]

    monthly_results[output_cols].to_csv(OUTPUT_FILE, index=False)
    microcap_diag.to_csv(OUTPUT_MICROCAP_DIAGNOSTICS, index=False)

    print(f"\nSaved monthly duration panel to: {Path(OUTPUT_FILE).resolve()}")
    print(f"Saved microcap diagnostics to: {Path(OUTPUT_MICROCAP_DIAGNOSTICS).resolve()}")
    print(f"Dropped invalid observations before outlier handling: {dropped_invalid_obs}/{total_obs}")
    print(f"Dropped duplicate firm-month rows: {dropped_dups}")

    if OUTLIER_METHOD == "drop":
        print(f"Dropped due to implied_duration outlier filter: {n_changed}")
    elif OUTLIER_METHOD == "winsorize":
        print(f"Winsorized implied_duration values changed: {n_changed}")

    if pd.notna(lower) and pd.notna(upper):
        print(f"Implied duration 1%/99% bounds: {lower:.6f}, {upper:.6f}")

    print(f"Final monthly observations saved: {len(monthly_results):,}")

    required_for_panel = ["gvkey", "month_date", "implied_duration_years"]
    print("\nPanel-compatibility check:")
    for c in required_for_panel:
        print(f"  {c} present: {c in monthly_results.columns}")
    print(monthly_results[required_for_panel].head(10))


if __name__ == "__main__":
    main()