"""Download daily market and risk-free return inputs through WRDS.

START_DATE and END_DATE select the sample. The output market_rf_daily.csv is
used by the event-return and asset-beta workflows. Access requires an authorized
WRDS account. Run from the project directory; the download executes on import.
"""

import time

import pandas as pd
import wrds

# ============================================================
# SETTINGS
# ============================================================
OUTPUT_FILE = "market_rf_daily.csv"

START_DATE = "1999-11-30"
END_DATE   = "2025-12-31"

# ============================================================
# HELPER
# ============================================================
def log(msg):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

# ============================================================
# CONNECT TO WRDS
# ============================================================
log("Connecting to WRDS...")
conn = wrds.Connection()
log("Connected.")

# ============================================================
# STEP 1: DOWNLOAD MARKET RETURN
# ============================================================
log("Downloading CRSP value-weighted market return...")

market = conn.raw_sql(f"""
    select
        date,
        vwretd
    from crsp.dsi
    where date between '{START_DATE}' and '{END_DATE}'
""", date_cols=["date"])

if market.empty:
    raise ValueError("Market return data download failed.")

market = market.rename(columns={
    "vwretd": "market_return"
})

log(f"Downloaded {len(market):,} market return rows.")

# ============================================================
# STEP 2: DOWNLOAD RISK-FREE RATE
# ============================================================
log("Downloading Fama-French daily risk-free rate...")

rf = conn.raw_sql(f"""
    select
        date,
        rf
    from ff.factors_daily
    where date between '{START_DATE}' and '{END_DATE}'
""", date_cols=["date"])

if rf.empty:
    raise ValueError("Risk-free rate download failed.")

rf = rf.rename(columns={
    "rf": "risk_free_rate"
})

log(f"Downloaded {len(rf):,} risk-free rate rows.")

# ============================================================
# STEP 3: MERGE MARKET RETURN AND RISK FREE
# ============================================================
log("Merging market return with risk-free rate...")

factors = pd.merge(
    market,
    rf,
    on="date",
    how="inner"
)

log(f"Merged dataset rows: {len(factors):,}")

# ============================================================
# STEP 4: COMPUTE EXCESS MARKET RETURN
# ============================================================
log("Computing excess market return...")

factors["market_excess"] = (
    factors["market_return"] - factors["risk_free_rate"]
)

# ============================================================
# STEP 5: SAVE
# ============================================================
log("Saving dataset...")

factors.to_csv(OUTPUT_FILE, index=False)

log(f"Saved to {OUTPUT_FILE}")

print(factors.head())