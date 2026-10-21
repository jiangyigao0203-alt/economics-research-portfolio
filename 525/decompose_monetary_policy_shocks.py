"""Decompose rate and equity surprises into monetary-policy and information shocks.

Read USMPD.csv with date_time, FF3, and SP500 columns. Search over orthogonal
rotations subject to opposite-sign monetary-policy and same-sign information
restrictions. The configured random seed controls the rotation search.
Run from the 525 directory; the workflow executes on import and writes a CSV.
"""

import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = "USMPD.csv"   # Source surprise-data file.
OUTPUT_FILE = "jk_shocks_ff3_sp500.csv"

DATE_COL = "date_time"
RATE_COL = "FF3"
STOCK_COL = "SP500"

# Optional: demean the reduced-form surprises before covariance estimation
# Demeaning estimates covariance from centered surprises.
DEMEAN = True

# Random rotation search
MAX_TRIES = 200000
RANDOM_SEED = 42

# Numerical tolerance for sign checks
TOL = 1e-12


# ============================================================
# HELPERS
# ============================================================
def random_orthogonal_2x2(rng):
    """
    Draw a random 2x2 orthogonal matrix Q using QR decomposition
    of a random normal matrix, following the usual RRWZ idea.
    """
    A = rng.standard_normal((2, 2))
    Q, R = np.linalg.qr(A)

    # Normalize signs so QR gives a proper random orthogonal draw
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q @ np.diag(diag_sign)

    return Q


def satisfies_jk_signs(H, tol=1e-12):
    """
    JK sign restrictions:
      column 1 = MP shock:    h11 > 0, h21 < 0
      column 2 = INFO shock:  h12 > 0, h22 > 0
    """
    h11, h12 = H[0, 0], H[0, 1]
    h21, h22 = H[1, 0], H[1, 1]

    return (
        (h11 > tol) and
        (h21 < -tol) and
        (h12 > tol) and
        (h22 > tol)
    )


def find_rotation(C, max_tries=100000, seed=42, tol=1e-12):
    """
    Given a Cholesky factor C such that Sigma = C C',
    search for an orthogonal Q such that H = C Q satisfies JK signs.
    """
    rng = np.random.default_rng(seed)

    for draw in range(1, max_tries + 1):
        Q = random_orthogonal_2x2(rng)
        H = C @ Q

        if satisfies_jk_signs(H, tol=tol):
            return H, Q, draw

    raise RuntimeError(
        f"No admissible rotation found after {max_tries} draws. "
        "Try increasing MAX_TRIES or inspect the data/sign convention."
    )


# ============================================================
# LOAD + CLEAN DATA
# ============================================================
df = pd.read_csv(INPUT_FILE)

# Parse date
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

# Keep only necessary columns
work = df[[DATE_COL, RATE_COL, STOCK_COL]].copy()

# Convert to numeric
for col in [RATE_COL, STOCK_COL]:
    work[col] = pd.to_numeric(work[col], errors="coerce")

# Drop missing
work = work.dropna(subset=[RATE_COL, STOCK_COL]).sort_values(DATE_COL).reset_index(drop=True)

if len(work) < 5:
    raise ValueError("Too few usable observations after cleaning.")

print(f"Usable observations: {len(work)}")
print(work.head())


# ============================================================
# BUILD REDUCED-FORM SURPRISES u_t
# ============================================================
U_raw = work[[RATE_COL, STOCK_COL]].to_numpy(dtype=float)

if DEMEAN:
    U_mean = U_raw.mean(axis=0, keepdims=True)
    U = U_raw - U_mean
else:
    U_mean = np.zeros((1, 2))
    U = U_raw.copy()

print("\nMeans removed from reduced-form surprises:")
print(pd.Series(U_mean.flatten(), index=[RATE_COL, STOCK_COL]))


# ============================================================
# COVARIANCE MATRIX Sigma_u
# ============================================================
Sigma_u = np.cov(U.T, ddof=1)

print("\nEstimated covariance matrix Sigma_u:")
print(pd.DataFrame(Sigma_u, index=[RATE_COL, STOCK_COL], columns=[RATE_COL, STOCK_COL]))

# Check positive definiteness
eigvals = np.linalg.eigvalsh(Sigma_u)
print("\nEigenvalues of Sigma_u:")
print(eigvals)

if np.any(eigvals <= 0):
    raise ValueError(
        "Sigma_u is not positive definite, so Cholesky failed in theory. "
        "Check data quality, duplicates, or near-collinearity."
    )


# ============================================================
# CHOLESKY STARTING FACTOR C
# Sigma_u = C C'
# ============================================================
C = np.linalg.cholesky(Sigma_u)

print("\nLower-triangular Cholesky factor C:")
print(pd.DataFrame(C, index=[RATE_COL, STOCK_COL], columns=["shock1", "shock2"]))


# ============================================================
# FIND ROTATION Q SO THAT H = C Q SATISFIES JK SIGNS
# ============================================================
H, Q, n_draws = find_rotation(C, max_tries=MAX_TRIES, seed=RANDOM_SEED, tol=TOL)

print(f"\nFound admissible rotation after {n_draws} draw(s).")

print("\nRotation matrix Q:")
print(pd.DataFrame(Q, index=["shock1", "shock2"], columns=["MP", "INFO"]))

print("\nImpact matrix H = C Q:")
print(pd.DataFrame(H, index=[RATE_COL, STOCK_COL], columns=["MP", "INFO"]))

print("\nCheck JK sign restrictions:")
print(f"h11 (rate response to MP)    = {H[0,0]: .6f}  > 0")
print(f"h21 (stock response to MP)   = {H[1,0]: .6f}  < 0")
print(f"h12 (rate response to INFO)  = {H[0,1]: .6f}  > 0")
print(f"h22 (stock response to INFO) = {H[1,1]: .6f}  > 0")


# ============================================================
# RECOVER STRUCTURAL SHOCKS
#
# Column-vector form:
#   u_t = H eps_t
#   eps_t = H^{-1} u_t
#
# If U stores observations row-wise, then:
#   eps_row = u_row @ (H^{-1})'
# ============================================================
H_inv = np.linalg.inv(H)
EPS = U @ H_inv.T

# Sanity check: reconstruct U from EPS
U_reconstructed = EPS @ H.T
max_abs_reconstruction_error = np.max(np.abs(U - U_reconstructed))

print(f"\nMax absolute reconstruction error: {max_abs_reconstruction_error:.12g}")


# ============================================================
# OUTPUT
# ============================================================
out = work[[DATE_COL]].copy()
out[f"u_{RATE_COL}"] = U[:, 0]
out[f"u_{STOCK_COL}"] = U[:, 1]
out["eps_MP"] = EPS[:, 0]
out["eps_INFO"] = EPS[:, 1]

# Retain the original undemeaned values for comparison.
out[f"{RATE_COL}_raw"] = U_raw[:, 0]
out[f"{STOCK_COL}_raw"] = U_raw[:, 1]

# Reconstructed surprises (for checking)
out[f"reconstructed_{RATE_COL}"] = U_reconstructed[:, 0]
out[f"reconstructed_{STOCK_COL}"] = U_reconstructed[:, 1]

out.to_csv(OUTPUT_FILE, index=False)

# Save matrices too
pd.DataFrame(
    Sigma_u,
    index=[RATE_COL, STOCK_COL],
    columns=[RATE_COL, STOCK_COL]
).to_csv("jk_sigma_u_ff3_sp500.csv")

pd.DataFrame(
    C,
    index=[RATE_COL, STOCK_COL],
    columns=["shock1", "shock2"]
).to_csv("jk_cholesky_C_ff3_sp500.csv")

pd.DataFrame(
    Q,
    index=["shock1", "shock2"],
    columns=["MP", "INFO"]
).to_csv("jk_rotation_Q_ff3_sp500.csv")

pd.DataFrame(
    H,
    index=[RATE_COL, STOCK_COL],
    columns=["MP", "INFO"]
).to_csv("jk_impact_H_ff3_sp500.csv")

print("\nSaved files:")
print(f" - {OUTPUT_FILE}")
print(" - jk_sigma_u_ff3_sp500.csv")
print(" - jk_cholesky_C_ff3_sp500.csv")
print(" - jk_rotation_Q_ff3_sp500.csv")
print(" - jk_impact_H_ff3_sp500.csv")