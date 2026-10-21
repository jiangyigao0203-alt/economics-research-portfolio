from pathlib import Path

import pandas as pd
import statsmodels.api as sm

# ============================================================
# SETTINGS
# ============================================================
INPUT_COEF_FILE = "panel_regression_coefficients.csv"
OUTPUT_RESULTS_FILE = "beta_monotonicity_test_results.csv"

# ============================================================
# LOAD COEFFICIENT TABLE
# ============================================================
df = pd.read_csv(INPUT_COEF_FILE)

required_cols = ["term", "coef", "std_err", "shock_type", "bin"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["coef", "std_err", "shock_type", "bin"]).copy()
df["bin"] = pd.to_numeric(df["bin"], errors="coerce")
df["coef"] = pd.to_numeric(df["coef"], errors="coerce")
df["std_err"] = pd.to_numeric(df["std_err"], errors="coerce")
df = df.dropna(subset=["bin", "coef", "std_err"]).copy()

# keep only positive std errors
df = df[df["std_err"] > 0].copy()

# ============================================================
# FUNCTION: TEST INCREASING TREND
# ============================================================
def test_increasing_beta(subdf, shock_name):
    subdf = subdf.sort_values("bin").copy()

    y = subdf["coef"].astype(float)
    X = sm.add_constant(subdf["bin"].astype(float))

    # inverse-variance weights
    w = 1.0 / (subdf["std_err"].astype(float) ** 2)

    model = sm.WLS(y, X, weights=w)
    results = model.fit()

    slope = results.params["bin"]
    slope_se = results.bse["bin"]
    t_stat = slope / slope_se

    # one-sided p-value for H1: slope > 0
    p_two_sided = results.pvalues["bin"]
    if slope >= 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - p_two_sided / 2

    out = {
        "shock_type": shock_name,
        "n_bins": len(subdf),
        "slope_on_bin": slope,
        "slope_std_err": slope_se,
        "t_stat": t_stat,
        "p_value_two_sided": p_two_sided,
        "p_value_one_sided_increasing": p_one_sided,
        "r_squared": results.rsquared,
        "intercept": results.params["const"],
    }

    return out, results

# ============================================================
# RUN TESTS
# ============================================================
all_rows = []
fitted_models = {}

for shock in ["MP", "INFO"]:
    sub = df[df["shock_type"] == shock].copy()
    if len(sub) < 3:
        print(f"Skipping {shock}: not enough bins.")
        continue

    row, res = test_increasing_beta(sub, shock)
    all_rows.append(row)
    fitted_models[shock] = res

results_df = pd.DataFrame(all_rows)
results_df.to_csv(OUTPUT_RESULTS_FILE, index=False)

print("\n===== MONOTONICITY TEST RESULTS =====\n")
print(results_df.to_string(index=False))

print(f"\nSaved results to: {Path(OUTPUT_RESULTS_FILE).resolve()}")

# ============================================================
# OPTIONAL: FULL REGRESSION SUMMARIES
# ============================================================
for shock, res in fitted_models.items():
    print(f"\n===== {shock} TREND REGRESSION =====\n")
    print(res.summary())