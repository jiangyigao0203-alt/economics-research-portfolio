"""Estimate firm-level abnormal returns around FOMC events.

Inputs are shock-event timestamps, matched CRSP prices, and market/risk-free
returns; filenames and event-window settings are defined below. Firm market
models use observations between consecutive events and forecast subsequent
returns. Outputs include event-level return estimates and a partial checkpoint.
Run from the 525 directory. This script executes its workflow on import.
"""

import logging
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ============================================================
# SETTINGS
# ============================================================
FOMC_FILE = "shocks_fed_jk_t.csv"   # must contain date_time, eps_MP, eps_INFO (and can contain other shock columns)
PRICE_FILE = "gvkey_crsp_daily_price_matched.csv"
MARKET_FILE = "market_rf_daily.csv"
OUTPUT_FILE = "forecasted_returns_5day_after_fomc_updated.csv"
INTERMEDIATE_FILE = "forecasted_returns_5day_after_fomc_updated_partial.csv"

FOMC_DATE_CANDIDATES = ["date_time", "Date", "date"]
PRICE_GVKEY_COL = "gvkey"
PRICE_DATE_COL = "date"
PRICE_COL = "price"

MARKET_DATE_COL = "date"
MARKET_RET_COL = "market_return"
RF_COL = "risk_free_rate"

T_FORECAST = 5
MIN_ESTIMATION_OBS = 20

# keep only columns that actually exist in the FOMC file
REQUESTED_SHOCK_COLS = [
    "pc1",
    "SP500",
    "eps_MP",
    "eps_INFO",
    "MP_median",
    "CBI_median",
]

# Progress update frequency
EVENT_PROGRESS_EVERY = 1
FIRM_PROGRESS_EVERY = 500
SAVE_INTERMEDIATE_EVERY = 10


# ============================================================
# HELPERS
# ============================================================
def log(msg: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)



def pick_first_existing(columns, candidates, label):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"Could not find {label}. Tried: {candidates}")



def safe_log_return(price_series: pd.Series) -> pd.Series:
    """Compute consecutive log-price differences while preserving the input index."""
    price_series = pd.to_numeric(price_series, errors="coerce")
    return pd.Series(np.log((price_series / price_series.shift(1)).to_numpy(dtype=float)),
                     index=price_series.index, name=price_series.name)



def fmt_seconds(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ============================================================
# LOAD FILES
# ============================================================
script_start = time.time()
log("Loading input files...")

fomc = pd.read_csv(FOMC_FILE)
prices = pd.read_csv(PRICE_FILE)
market = pd.read_csv(MARKET_FILE)

fomc_date_col = pick_first_existing(fomc.columns, FOMC_DATE_CANDIDATES, "FOMC date column")

fomc[fomc_date_col] = pd.to_datetime(fomc[fomc_date_col], errors="coerce")
prices[PRICE_DATE_COL] = pd.to_datetime(prices[PRICE_DATE_COL], errors="coerce")
market[MARKET_DATE_COL] = pd.to_datetime(market[MARKET_DATE_COL], errors="coerce")

fomc = fomc.dropna(subset=[fomc_date_col]).copy()
prices = prices.dropna(subset=[PRICE_GVKEY_COL, PRICE_DATE_COL, PRICE_COL]).copy()
market = market.dropna(subset=[MARKET_DATE_COL, MARKET_RET_COL, RF_COL]).copy()

prices[PRICE_GVKEY_COL] = prices[PRICE_GVKEY_COL].astype(str).str.zfill(6)

# EVENT-DATE ALIGNMENT:
# the new FOMC file has intraday timestamps (e.g. 1990-02-08 11:30), while
# CRSP daily prices are date-level. Normalize both to calendar dates so that
# same-day observations are matched on the trading date.
fomc["event_datetime"] = fomc[fomc_date_col]
fomc["event_date"] = fomc[fomc_date_col].dt.normalize()
prices[PRICE_DATE_COL] = prices[PRICE_DATE_COL].dt.normalize()
market[MARKET_DATE_COL] = market[MARKET_DATE_COL].dt.normalize()

fomc = fomc.sort_values("event_datetime").reset_index(drop=True)
prices = prices.sort_values([PRICE_GVKEY_COL, PRICE_DATE_COL]).reset_index(drop=True)
market = market.sort_values(MARKET_DATE_COL).reset_index(drop=True)

shock_cols = [c for c in REQUESTED_SHOCK_COLS if c in fomc.columns]
if "eps_MP" not in shock_cols or "eps_INFO" not in shock_cols:
    raise ValueError("FOMC file must contain eps_MP and eps_INFO.")

# If there are multiple intraday rows on the same calendar date, keep the last one.
fomc_daily = (
    fomc[["event_datetime", "event_date"] + shock_cols]
    .sort_values("event_datetime")
    .drop_duplicates(subset=["event_date"], keep="last")
    .reset_index(drop=True)
)

log(f"Loaded raw FOMC rows: {len(fomc):,}")
log(f"Loaded daily FOMC event dates: {len(fomc_daily):,}")
log(f"Loaded price rows: {len(prices):,}")
log(f"Loaded market/rf rows: {len(market):,}")
log(f"Using shock columns: {shock_cols}")

# ============================================================
# PREPARE DAILY STOCK RETURNS
# ============================================================
log("Computing firm daily log returns...")
prices["stock_return"] = (
    prices.groupby(PRICE_GVKEY_COL, group_keys=False)[PRICE_COL]
    .apply(safe_log_return)
)

log("Merging daily prices with market return and risk-free rate...")
daily = prices.merge(
    market[[MARKET_DATE_COL, MARKET_RET_COL, RF_COL]],
    left_on=PRICE_DATE_COL,
    right_on=MARKET_DATE_COL,
    how="left"
)

daily = daily.dropna(subset=["stock_return", MARKET_RET_COL, RF_COL]).copy()
daily["stock_excess"] = daily["stock_return"] - daily[RF_COL]
daily["market_excess"] = daily[MARKET_RET_COL] - daily[RF_COL]

log(f"Daily usable rows after merge: {len(daily):,}")

# ============================================================
# BUILD EVENT LIST
# ============================================================
events = fomc_daily.copy()
events["prev_event_date"] = events["event_date"].shift(1)
events = events.dropna(subset=["prev_event_date"]).reset_index(drop=True)

n_events = len(events)
gvkeys = sorted(daily[PRICE_GVKEY_COL].unique().tolist())
n_firms = len(gvkeys)

log(f"Usable FOMC events with prior event: {n_events:,}")
log(f"Usable firms: {n_firms:,}")

# Optional speed-up: pre-split by firm once
log("Pre-splitting daily panel by firm for faster looping...")
firm_data = {gvkey: g.sort_values(PRICE_DATE_COL).copy() for gvkey, g in daily.groupby(PRICE_GVKEY_COL)}
log(f"Prepared firm dictionary with {len(firm_data):,} firms.")

# ============================================================
# MAIN LOOP
# ============================================================
results = []
completed_event_count = 0
overall_firm_evaluations = 0
overall_successful_forecasts = 0
loop_start = time.time()

log("Starting main CAPM forecasting loop...")

for event_idx, (_, ev) in enumerate(events.iterrows()):
    event_start = time.time()

    event_date = ev["event_date"]
    event_datetime = ev["event_datetime"]
    prev_event_date = ev["prev_event_date"]

    event_successes = 0
    event_firm_checks = 0

    if (event_idx + 1) % EVENT_PROGRESS_EVERY == 0:
        elapsed = time.time() - loop_start
        avg_per_event = elapsed / max(event_idx, 1) if event_idx > 0 else 0
        remaining_events = n_events - event_idx
        eta = avg_per_event * remaining_events if event_idx > 0 else np.nan

        log(
            f"Starting event {event_idx + 1}/{n_events} | "
            f"event_date={event_date.date()} | "
            f"previous_event={prev_event_date.date()} | "
            f"elapsed={fmt_seconds(elapsed)} | "
            f"ETA={fmt_seconds(eta) if event_idx > 0 else 'estimating...'}"
        )

    for firm_idx, gvkey in enumerate(gvkeys, start=1):
        event_firm_checks += 1
        overall_firm_evaluations += 1

        if firm_idx % FIRM_PROGRESS_EVERY == 0:
            elapsed = time.time() - event_start
            avg_per_firm = elapsed / firm_idx
            firms_left = n_firms - firm_idx
            eta_event = avg_per_firm * firms_left

            log(
                f"  Event {event_idx + 1}/{n_events}: "
                f"processed firm {firm_idx}/{n_firms} | "
                f"current gvkey={gvkey} | "
                f"event successes={event_successes:,} | "
                f"event ETA={fmt_seconds(eta_event)}"
            )

        firm = firm_data.get(gvkey)
        if firm is None or firm.empty:
            continue

        # Estimation window: (prev_event_date, event_date)
        est = firm[
            (firm[PRICE_DATE_COL] > prev_event_date) &
            (firm[PRICE_DATE_COL] < event_date)
        ][[PRICE_DATE_COL, "stock_excess", "market_excess"]].dropna()

        if len(est) < MIN_ESTIMATION_OBS:
            continue

        X = sm.add_constant(est["market_excess"])
        y = est["stock_excess"]

        try:
            model = sm.OLS(y, X).fit()
        except Exception:
            logging.getLogger(__name__).exception("Operation failed in %s", __file__)
            continue

        alpha_hat = model.params.get("const", np.nan)
        beta_hat = model.params.get("market_excess", np.nan)

        if pd.isna(alpha_hat) or pd.isna(beta_hat):
            continue

        # Forecast window: first 5 trading days on/after the calendar event date.
        # If there are not enough price observations, discard this firm-event.
        fut = firm[
            firm[PRICE_DATE_COL] >= event_date
        ][[PRICE_DATE_COL, "stock_return", MARKET_RET_COL, RF_COL, "market_excess"]].dropna()

        if len(fut) < T_FORECAST:
            continue

        fut = fut.sort_values(PRICE_DATE_COL).head(T_FORECAST).copy()

        fut["forecast_return"] = fut[RF_COL] + alpha_hat + beta_hat * fut["market_excess"]
        fut["abnormal_return"] = fut["stock_return"] - fut["forecast_return"]
        fut["car_through_day"] = fut["abnormal_return"].cumsum()

        fut["gvkey"] = gvkey
        fut["event_datetime"] = event_datetime
        fut["event_date"] = event_date
        fut["prev_event_date"] = prev_event_date
        fut["alpha_hat"] = alpha_hat
        fut["beta_hat"] = beta_hat
        fut["estimation_obs"] = len(est)
        fut["forecast_horizon_day"] = np.arange(len(fut))

        for col in shock_cols:
            fut[col] = ev[col]

        results.append(
            fut[[
                "gvkey",
                "event_datetime",
                "event_date",
                "prev_event_date",
                PRICE_DATE_COL,
                "forecast_horizon_day",
                "stock_return",
                "forecast_return",
                "abnormal_return",
                "car_through_day",
                MARKET_RET_COL,
                RF_COL,
                "alpha_hat",
                "beta_hat",
                "estimation_obs",
            ] + shock_cols].rename(columns={PRICE_DATE_COL: "trade_date"})
        )

        event_successes += 1
        overall_successful_forecasts += 1

    completed_event_count += 1
    event_elapsed = time.time() - event_start
    total_elapsed = time.time() - loop_start
    avg_per_event = total_elapsed / completed_event_count
    remaining_events = n_events - completed_event_count
    total_eta = avg_per_event * remaining_events

    log(
        f"Finished event {event_idx + 1}/{n_events} | "
        f"event_date={event_date.date()} | "
        f"firms checked={event_firm_checks:,} | "
        f"successful forecasts={event_successes:,} | "
        f"event runtime={fmt_seconds(event_elapsed)} | "
        f"total elapsed={fmt_seconds(total_elapsed)} | "
        f"total ETA={fmt_seconds(total_eta)}"
    )

    if completed_event_count % SAVE_INTERMEDIATE_EVERY == 0 and len(results) > 0:
        partial = pd.concat(results, ignore_index=True)
        partial = partial.sort_values(["event_date", "gvkey", "trade_date"]).reset_index(drop=True)
        partial.to_csv(INTERMEDIATE_FILE, index=False)
        log(
            f"Intermediate save completed after {completed_event_count} events: "
            f"{INTERMEDIATE_FILE} | rows={len(partial):,}"
        )

# ============================================================
# SAVE OUTPUT
# ============================================================
if len(results) == 0:
    raise ValueError(
        "No forecast results were generated. Check date overlap, column names, and whether enough post-event trading days exist."
    )

log("Concatenating final results...")
out = pd.concat(results, ignore_index=True)
out = out.sort_values(["event_date", "gvkey", "trade_date"]).reset_index(drop=True)

log("Saving final output...")
out.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# FINAL SUMMARY
# ============================================================
total_runtime = time.time() - script_start
log("All steps completed successfully.")
log(f"Total events processed: {completed_event_count:,}")
log(f"Total firm evaluations: {overall_firm_evaluations:,}")
log(f"Total successful firm-event forecasts: {overall_successful_forecasts:,}")
log(f"Final output rows: {len(out):,}")
log(f"Saved final output to: {OUTPUT_FILE}")
log(f"Total runtime: {fmt_seconds(total_runtime)}")

print(out.head(20))
