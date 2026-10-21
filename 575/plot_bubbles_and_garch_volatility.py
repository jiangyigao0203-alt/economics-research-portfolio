"""Plot bubble detections alongside manually estimated GARCH(1,1) volatility.

Read detection.csv containing date, price, bsadf, critical_value, and is_bubble.
Estimate conditional variance from price returns and plot prices, test
statistics, and volatility with bubble overlays. Review annualization settings
for the market's observation frequency. Run from the 575 directory.
Estimation, file output, and plotting execute on import.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =========================================================
# SETTINGS
# =========================================================
INPUT_FILE = Path("detection.csv")   # Merged detection statistics and bubble labels.
OUTPUT_PLOT_FILE = Path("bubble_with_garch_vol.png")

DATE_COL = "date"
PRICE_COL = "price"
BSADF_COL = "bsadf"
CV_COL = "critical_value"
BUBBLE_COL = "is_bubble"

SERIES_NAME = "Bubble Identification with Estimated Volatility"
PRICE_LABEL = "Price"
BSADF_LABEL = "BSADF"
CV_LABEL = "Monte Carlo Critical Value"
VOL_LABEL = "Estimated Volatility (GARCH)"
DATE_LABEL = "Date"
STAT_LABEL = "Statistic"

ANNUALIZATION_FACTOR = 365   # 365 for crypto daily, 252 for stock daily
ANNUALIZE_VOL = True

# =========================================================
# GARCH(1,1) FUNCTIONS
# =========================================================
def garch_filter(params, returns):
    """
    GARCH(1,1):
        sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    """
    omega, alpha, beta = params
    n = len(returns)
    variances = np.zeros(n)

    # initialize with sample variance
    variances[0] = np.var(returns)

    for t in range(1, n):
        variances[t] = omega + alpha * (returns[t - 1] ** 2) + beta * variances[t - 1]

        # safety floor
        variances[t] = max(1e-12, variances[t])

    return variances


def garch_log_likelihood(params, returns):
    """Return the Gaussian GARCH negative log-likelihood used by numerical minimization."""
    omega, alpha, beta = params

    # parameter restrictions
    if omega <= 1e-8 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return 1e12

    variances = garch_filter(params, returns)

    # extra safety
    if np.any(variances <= 0) or np.any(~np.isfinite(variances)):
        return 1e12

    log_lik = -0.5 * np.sum(
        np.log(2 * np.pi * variances) + (returns ** 2) / variances
    )

    return -log_lik


# =========================================================
# BUBBLE OVERLAY HELPER
# =========================================================
def overlay_bubble_segments(ax, x, y, is_bubble, linewidth=2.2, color="red"):
    """Highlight contiguous bubble-labelled segments without joining separated episodes."""
    if len(y) < 2:
        return

    bubble_seg = is_bubble[:-1] & is_bubble[1:]
    start_idx = None
    n_seg = len(bubble_seg)

    for i in range(n_seg):
        if bubble_seg[i] and start_idx is None:
            start_idx = i
        elif not bubble_seg[i] and start_idx is not None:
            ax.plot(x[start_idx:i + 1], y[start_idx:i + 1], linewidth=linewidth, color=color)
            start_idx = None

    if start_idx is not None:
        ax.plot(x[start_idx:n_seg + 1], y[start_idx:n_seg + 1], linewidth=linewidth, color=color)


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_FILE)

required_cols = [DATE_COL, PRICE_COL, BSADF_COL, CV_COL, BUBBLE_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in detection file: {missing}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)
df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
df[BSADF_COL] = pd.to_numeric(df[BSADF_COL], errors="coerce")
df[CV_COL] = pd.to_numeric(df[CV_COL], errors="coerce")

# robust boolean conversion
if df[BUBBLE_COL].dtype != bool:
    df[BUBBLE_COL] = df[BUBBLE_COL].astype(str).str.lower().map(
        {"true": True, "false": False}
    )

df = df.dropna(subset=[DATE_COL, PRICE_COL]).sort_values(DATE_COL).reset_index(drop=True)
df = df[df[PRICE_COL] > 0].copy()

if len(df) < 20:
    raise ValueError("Not enough valid price observations.")

# =========================================================
# COMPUTE RETURNS
# =========================================================
df["log_price"] = np.log(df[PRICE_COL])
df["log_return"] = df["log_price"].diff()

df_ret = df.dropna(subset=["log_return"]).copy().reset_index(drop=True)

if len(df_ret) < 20:
    raise ValueError("Not enough return observations to estimate GARCH.")

# =========================================================
# ESTIMATE GARCH VOLATILITY
# =========================================================
returns_raw = df_ret["log_return"].to_numpy(dtype=float)
returns_scaled = returns_raw * 100.0   # for numerical stability

v = np.var(returns_scaled)
initial_params = [max(v * 0.05, 1e-6), 0.05, 0.90]

opt_result = minimize(
    garch_log_likelihood,
    initial_params,
    args=(returns_scaled,),
    bounds=((1e-8, None), (0, 1), (0, 1)),
    method="SLSQP"
)

if not opt_result.success:
    print("WARNING: Optimization did not fully converge.")
    print("Message:", opt_result.message)

optimized_params = opt_result.x
cond_var_scaled = garch_filter(optimized_params, returns_scaled)

df_ret["garch_vol"] = np.sqrt(cond_var_scaled) / 100.0

if ANNUALIZE_VOL:
    df_ret["garch_vol"] = df_ret["garch_vol"] * np.sqrt(ANNUALIZATION_FACTOR)

# merge back
df = df.merge(df_ret[[DATE_COL, "garch_vol"]], on=DATE_COL, how="left")

# =========================================================
# PRINT RESULTS
# =========================================================
print("-" * 40)
print("GARCH(1,1) RESULTS")
print("-" * 40)
print(f"Omega (ω): {optimized_params[0]:.6f}")
print(f"Alpha (α): {optimized_params[1]:.4f}")
print(f"Beta  (β): {optimized_params[2]:.4f}")
print(f"Persistence (α+β): {optimized_params[1] + optimized_params[2]:.4f}")
print(f"Converged: {opt_result.success}")
print("-" * 40)

# =========================================================
# PLOT
# =========================================================
x = df[DATE_COL].to_numpy()
price = df[PRICE_COL].to_numpy()
bsadf = df[BSADF_COL].to_numpy()
cv = df[CV_COL].to_numpy()
is_bubble = df[BUBBLE_COL].fillna(False).to_numpy()
garch_vol = df["garch_vol"].to_numpy()

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1,
    figsize=(14, 12),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1, 1]}
)

# ---------------------------------------------------------
# Panel 1: Price with bubbles highlighted
# ---------------------------------------------------------
ax1.plot(x, price, linewidth=1.2, label=PRICE_LABEL, color="blue")
overlay_bubble_segments(ax1, x, price, is_bubble, linewidth=2.2, color="red")
ax1.set_title(f"{SERIES_NAME}")
ax1.set_ylabel("Price")
ax1.grid(True, alpha=0.3)
ax1.legend()

# ---------------------------------------------------------
# Panel 2: BSADF vs Critical Value
# ---------------------------------------------------------
ax2.plot(x, bsadf, linewidth=1.2, label=BSADF_LABEL, color="blue")
ax2.plot(x, cv, linewidth=1.2, label=CV_LABEL, color="orange")
overlay_bubble_segments(ax2, x, bsadf, is_bubble, linewidth=2.2, color="red")
ax2.set_title("Bubble Identification Graph")
ax2.set_ylabel(STAT_LABEL)
ax2.grid(True, alpha=0.3)
ax2.legend()

# ---------------------------------------------------------
# Panel 3: Estimated Volatility
# ---------------------------------------------------------
ax3.plot(x, garch_vol, linewidth=1.2, label=VOL_LABEL, color="green")
overlay_bubble_segments(ax3, x, garch_vol, is_bubble, linewidth=2.2, color="red")
ax3.set_title("Estimated Volatility (GARCH 1,1)")
ax3.set_ylabel("Volatility")
ax3.set_xlabel(DATE_LABEL)
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_FILE, dpi=300)
plt.show()

print(f"Saved plot to: {OUTPUT_PLOT_FILE}")