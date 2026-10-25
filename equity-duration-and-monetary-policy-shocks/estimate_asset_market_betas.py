"""Estimate asset market betas from matched daily return observations.

Read the configured firm-price and market/risk-free CSV inputs, construct
returns, and estimate covariance-to-variance slopes by firm. Export beta
estimates and sample-quality diagnostics. Run from the project directory.
The calculation and CSV exports execute on import.
"""

import logging

import numpy as np
import pandas as pd

# =========================
# SETTINGS
# =========================
FILE_MARKET = "market_rf_daily.csv"
FILE_PRICE  = "gvkey_crsp_daily_price_matched.csv"

ID_COL = "gvkey"   # Alternative identifier: "permno".
MIN_OBS = 30       # minimum observations required for beta

# =========================
# STEP 1: LOAD DATA
# =========================
print("STEP 1: Loading data...")

market = pd.read_csv(FILE_MARKET)
price  = pd.read_csv(FILE_PRICE)

# Parse dates
market["date"] = pd.to_datetime(market["date"], errors="coerce")
price["date"]  = pd.to_datetime(price["date"], errors="coerce")

# Drop bad rows early
market = market.dropna(subset=["date", "market_excess"])
price  = price.dropna(subset=[ID_COL, "date", "price"])

print(f"Market rows: {len(market):,}")
print(f"Price rows:  {len(price):,}")

# =========================
# STEP 2: SORT & COMPUTE LOG RETURNS
# =========================
print("\nSTEP 2: Computing stock returns...")

price = price.sort_values([ID_COL, "date"]).copy()

# Ensure positive prices
price = price[price["price"] > 0].copy()

# Log return
price["log_price"] = np.log(price["price"])
price["stock_return"] = price.groupby(ID_COL)["log_price"].diff()

# =========================
# STEP 3: MERGE WITH MARKET
# =========================
print("\nSTEP 3: Merging with market data...")

df = price.merge(
    market[["date", "market_excess"]],
    on="date",
    how="inner"
)

# Drop missing returns
df = df.dropna(subset=["stock_return", "market_excess"])

print(f"Total merged rows: {len(df):,}")

# =========================
# STEP 4: COMPUTE BETA
# =========================
print("\nSTEP 4: Computing betas...")

def compute_beta(group):
    """Return the covariance-to-market-variance slope, observation count, and validity flag for one firm."""
    Ri = group["stock_return"]
    Rm = group["market_excess"]

    n_obs = len(group)

    # Not enough data
    if n_obs < MIN_OBS:
        return pd.Series({
            "beta": np.nan,
            "n_obs": n_obs,
            "reason": "too_few_obs"
        })

    var = np.var(Rm, ddof=1)

    if var == 0 or not np.isfinite(var):
        return pd.Series({
            "beta": np.nan,
            "n_obs": n_obs,
            "reason": "zero_var_market"
        })

    cov = np.cov(Ri, Rm, ddof=1)[0, 1]

    beta = cov / var

    if not np.isfinite(beta):
        return pd.Series({
            "beta": np.nan,
            "n_obs": n_obs,
            "reason": "invalid_beta"
        })

    return pd.Series({
        "beta": beta,
        "n_obs": n_obs,
        "reason": "ok"
    })

beta_df = (
    df.groupby(ID_COL)
      .apply(compute_beta)
      .reset_index()
)

# =========================
# STEP 5: DIAGNOSTICS
# =========================
print("\nSTEP 5: BETA DIAGNOSTICS")

total_firms = beta_df[ID_COL].nunique()
valid_beta = beta_df["beta"].notna().sum()
missing_beta = beta_df["beta"].isna().sum()

print("\n========== SUMMARY ==========")
print(f"Total firms:            {total_firms:,}")
print(f"Firms with beta:        {valid_beta:,}")
print(f"Firms WITHOUT beta:     {missing_beta:,}")
print(f"Coverage:               {valid_beta / total_firms:.2%}")

# Breakdown of missing reasons
print("\n========== MISSING BREAKDOWN ==========")

missing_reasons = (
    beta_df[beta_df["beta"].isna()]
    .groupby("reason")
    .size()
    .reset_index(name="count")
)

if len(missing_reasons) > 0:
    for _, row in missing_reasons.iterrows():
        print(f"{row['reason']:20s}: {row['count']:,}")
else:
    print("No missing betas!")

# Distribution of observations
print("\n========== OBSERVATION STATS ==========")
print(beta_df["n_obs"].describe())

# =========================
# STEP 6: CLEAN OUTPUT
# =========================
beta_df_clean = beta_df.dropna(subset=["beta"]).copy()

print(f"\nFinal usable betas: {len(beta_df_clean):,}")

# =========================
# STEP 7: SAVE OUTPUTS
# =========================
OUTPUT_FILE = "asset_betas.csv"
beta_df_clean[[ID_COL, "beta"]].to_csv(OUTPUT_FILE, index=False)

# Optional: save full diagnostics
beta_df.to_csv("asset_betas_diagnostics.csv", index=False)

print(f"\nSaved cleaned betas to: {OUTPUT_FILE}")
print("Saved full diagnostics to: asset_betas_diagnostics.csv")

# =========================
# STEP 8: (OPTIONAL) PANEL COVERAGE CHECK
# =========================
print("\nSTEP 8: (Optional) Panel coverage check")

# Optional coverage check when an analysis panel is available.
try:
    panel = pd.read_csv("panel_regression_ready.csv")

    panel_firms = set(panel[ID_COL].astype(str))
    beta_firms  = set(beta_df_clean[ID_COL].astype(str))

    overlap = len(panel_firms & beta_firms)

    print("\n========== PANEL COVERAGE ==========")
    print(f"Panel firms:           {len(panel_firms):,}")
    print(f"Beta firms:            {len(beta_firms):,}")
    print(f"Overlap:               {overlap:,}")
    print(f"Coverage in panel:     {overlap / len(panel_firms):.2%}")

except Exception:
    logging.getLogger(__name__).exception("Operation failed in %s", __file__)
    print("Panel file not found or skipped.")
