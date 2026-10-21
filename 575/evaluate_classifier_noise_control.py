"""Run a chronological classifier control with a random-noise feature.

Read the bubble/volatility panel, split each series by time, and replace the
volatility feature with seeded random noise. Fit the logistic/threshold model
and export performance metrics and test-probability plots. This is a diagnostic
control rather than the main volatility-based predictive specification.
Review INPUT_PANEL_FILE and output paths before running.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# CONFIG
# =========================================================

INPUT_PANEL_FILE = Path(__file__).resolve().parent / 'dataset_build_outputs' / '111' / 'panel_dataset_full.csv'
OUTPUT_CSV       = Path(__file__).resolve().parent / 'mle_model_outputs' / 'single_run_time_split_random_noise_results.csv'

# New plot outputs
OUTPUT_PLOT_ALL_TEST_PROBS = Path(__file__).resolve().parent / 'mle_model_outputs' / 'test_probability_distribution_all.png'
OUTPUT_PLOT_TEST_BY_CLASS  = Path(__file__).resolve().parent / 'mle_model_outputs' / 'test_probability_distribution_by_true_class.png'
OUTPUT_PLOT_TEST_BOX       = Path(__file__).resolve().parent / 'mle_model_outputs' / 'test_probability_boxplot_by_true_class.png'

TEST_SIZE = 0.30

DATE_COL   = "date"
SERIES_COL = "series_name"
TARGET_COL = "is_bubble"

# IMPORTANT: lagged vol is replaced by random noise
USE_RANDOM_NOISE_FEATURE      = True
RANDOM_NOISE_SEED             = 42
INCLUDE_MARKET_TYPE_DUMMIES   = True

USE_NOISE_THRESHOLD_MODEL     = True
THRESHOLD_GRID_GAMMA0         = np.linspace(-4.0, 4.0, 81)
THRESHOLD_GRID_GAMMA1         = np.linspace(-4.0, 4.0, 81)
THRESHOLD_REFINE_ROUNDS       = 2
THRESHOLD_REFINE_STEP         = 0.5
THRESHOLD_REFINE_POINTS_PER_SIDE = 10

FALLBACK_THRESHOLD = 0.50
MAX_ITER           = 50
TOL                = 1e-8
RIDGE              = 1e-8

# Plot style
PLOT_BINS = 50
PLOT_DPI  = 180


# =========================================================
# HELPERS
# =========================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def normalize_target(s):
    """Map supported boolean/string bubble labels to a binary numeric target."""
    if s.dtype == bool:
        return s.fillna(False).astype(int)
    return (
        s.astype(str).str.strip().str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0).astype(int)
    )


def compute_auc(y_true, p_pred):
    y = np.asarray(y_true, int)
    p = np.asarray(p_pred, float)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = pd.Series(p).rank(method="average").values[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_f1(y_true, y_hat):
    """Compute the harmonic mean of precision and recall, handling empty positive predictions."""
    y = np.asarray(y_true, int)
    h = np.asarray(y_hat, int)
    tp = int(((y == 1) & (h == 1)).sum())
    fp = int(((y == 0) & (h == 1)).sum())
    fn = int(((y == 1) & (h == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def confusion(y_true, y_hat):
    y = np.asarray(y_true, int)
    h = np.asarray(y_hat, int)
    tp = int(((y == 1) & (h == 1)).sum())
    tn = int(((y == 0) & (h == 0)).sum())
    fp = int(((y == 0) & (h == 1)).sum())
    fn = int(((y == 1) & (h == 0)).sum())
    return tp, tn, fp, fn


def metrics(y_true, y_hat, p_pred):
    tp, tn, fp, fn = confusion(y_true, y_hat)
    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = compute_f1(y_true, y_hat)
    auc = compute_auc(y_true, p_pred)
    pred_pos_rate = float(np.mean(y_hat)) if n > 0 else np.nan
    return {
        "n": int(n),
        "positive_rate": float(np.mean(y_true)) if n > 0 else np.nan,
        "predicted_positive_rate": pred_pos_rate,
        "auc": auc,
        "accuracy": float(acc) if n > 0 else np.nan,
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def plot_test_probability_distributions(y_test, p_test):
    """
    Make plots to visually check whether predicted probabilities are systematically reversed
    or poorly separated on the TEST set.
    """
    y = np.asarray(y_test, int)
    p = np.asarray(p_test, float)

    p_pos = p[y == 1]
    p_neg = p[y == 0]

    # -----------------------------------------------------
    # Plot 1: Overall distribution
    # -----------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(p, bins=PLOT_BINS, edgecolor="black")
    plt.title("Test Set: Distribution of Predicted Probabilities")
    plt.xlabel("Predicted probability")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_ALL_TEST_PROBS, dpi=PLOT_DPI)
    plt.close()

    # -----------------------------------------------------
    # Plot 2: Distribution by true class
    # -----------------------------------------------------
    plt.figure(figsize=(8, 5))
    if len(p_neg) > 0:
        plt.hist(p_neg, bins=PLOT_BINS, alpha=0.6, label="True class = 0", density=False)
    if len(p_pos) > 0:
        plt.hist(p_pos, bins=PLOT_BINS, alpha=0.6, label="True class = 1", density=False)

    plt.title("Test Set: Predicted Probabilities by True Class")
    plt.xlabel("Predicted probability")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_TEST_BY_CLASS, dpi=PLOT_DPI)
    plt.close()

    # -----------------------------------------------------
    # Plot 3: Boxplot by true class
    # -----------------------------------------------------
    box_data = []
    box_labels = []
    if len(p_neg) > 0:
        box_data.append(p_neg)
        box_labels.append("True 0")
    if len(p_pos) > 0:
        box_data.append(p_pos)
        box_labels.append("True 1")

    if box_data:
        plt.figure(figsize=(6, 5))
        plt.boxplot(box_data, tick_labels=box_labels)
        plt.title("Test Set: Predicted Probability Boxplot by True Class")
        plt.ylabel("Predicted probability")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_TEST_BOX, dpi=PLOT_DPI)
        plt.close()

    # -----------------------------------------------------
    # Console summary for quick diagnosis
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST PROBABILITY DISTRIBUTION SUMMARY")
    print("=" * 80)

    def summarize(arr, name):
        if len(arr) == 0:
            print(f"{name}: no observations")
            return
        print(f"{name}:")
        print(f"  n      = {len(arr)}")
        print(f"  mean   = {np.mean(arr):.6f}")
        print(f"  median = {np.median(arr):.6f}")
        print(f"  min    = {np.min(arr):.6f}")
        print(f"  p25    = {np.quantile(arr, 0.25):.6f}")
        print(f"  p75    = {np.quantile(arr, 0.75):.6f}")
        print(f"  max    = {np.max(arr):.6f}")

    summarize(p, "All test probabilities")
    summarize(p_neg, "True class = 0")
    summarize(p_pos, "True class = 1")

    if len(p_pos) > 0 and len(p_neg) > 0:
        print("\nInterpretation check:")
        print(f"  mean(prob | true=1) = {np.mean(p_pos):.6f}")
        print(f"  mean(prob | true=0) = {np.mean(p_neg):.6f}")
        if np.mean(p_pos) < np.mean(p_neg):
            print("  WARNING: positives receive LOWER probabilities on average than negatives.")
            print("  This suggests systematic reversal / misranking.")
        else:
            print("  Positives receive HIGHER probabilities on average than negatives.")
            print("  This is directionally sensible.")


# =========================================================
# DATA
# =========================================================

def load_data():
    df = pd.read_csv(INPUT_PANEL_FILE)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[DATE_COL, SERIES_COL, TARGET_COL]).copy()
    df[TARGET_COL] = normalize_target(df[TARGET_COL])

    if USE_RANDOM_NOISE_FEATURE:
        rng = np.random.default_rng(RANDOM_NOISE_SEED)
        df["random_noise"] = rng.normal(loc=0.0, scale=1.0, size=len(df))

    x_cols = []
    if USE_RANDOM_NOISE_FEATURE:
        x_cols.append("random_noise")

    if INCLUDE_MARKET_TYPE_DUMMIES:
        dummies = pd.get_dummies(df["market_type"], prefix="market_type", drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        x_cols.extend(dummies.columns.tolist())

    keep = [DATE_COL, SERIES_COL, "market_type", TARGET_COL, "random_noise"] \
           + [c for c in x_cols if c != "random_noise"]

    df = df[keep].dropna(subset=[TARGET_COL] + x_cols).reset_index(drop=True)

    print(f"Loaded {len(df)} rows | {df[SERIES_COL].nunique()} series | positive rate = {df[TARGET_COL].mean():.4f}")
    return df, x_cols


def time_split_by_series(df):
    """Split each series chronologically and standardize features using training observations."""
    train_parts, test_parts = [], []

    for _, g in df.groupby(SERIES_COL, sort=False):
        g = g.sort_values(DATE_COL).copy()
        n = len(g)
        if n < 5:
            train_parts.append(g)
        else:
            split_idx = int(np.floor(n * (1 - TEST_SIZE)))
            split_idx = min(max(split_idx, 1), n - 1)
            train_parts.append(g.iloc[:split_idx].copy())
            test_parts.append(g.iloc[split_idx:].copy())

    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    cont = ["random_noise"] if USE_RANDOM_NOISE_FEATURE else []
    train, test = train.copy(), test.copy()
    for c in cont:
        mu = train[c].mean()
        sd = train[c].std()
        if pd.isna(sd) or sd == 0:
            sd = 1.0
        train[c] = (train[c] - mu) / sd
        test[c] = (test[c] - mu) / sd

    return train, test


# =========================================================
# MODEL
# =========================================================

def fit_irls(X, y):
    """Fit logistic coefficients with a stabilized iteratively reweighted least-squares update."""
    k = X.shape[1]
    beta = np.zeros(k)
    for _ in range(MAX_ITER):
        p = np.clip(sigmoid(X @ beta), 1e-8, 1 - 1e-8)
        grad = X.T @ (y - p)
        H = X.T @ (X * (p * (1 - p))[:, None]) + RIDGE * np.eye(k)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        beta += step
        if np.abs(step).max() < TOL:
            break
    return beta


def find_best_threshold(y, p, noise_feature):
    """Select the configured threshold parameters using training-sample F1 for the noise control."""
    best_f1, best_g0, best_g1 = -np.inf, 0.0, 0.0

    def search(g0s, g1s):
        nonlocal best_f1, best_g0, best_g1
        for g0 in g0s:
            for g1 in g1s:
                thr = sigmoid(g0 + g1 * noise_feature)
                y_hat = (p >= thr).astype(int)
                f1 = compute_f1(y, y_hat)
                if f1 > best_f1:
                    best_f1, best_g0, best_g1 = f1, float(g0), float(g1)

    search(THRESHOLD_GRID_GAMMA0, THRESHOLD_GRID_GAMMA1)

    step = THRESHOLD_REFINE_STEP
    pts = THRESHOLD_REFINE_POINTS_PER_SIDE
    for _ in range(THRESHOLD_REFINE_ROUNDS):
        search(
            np.linspace(best_g0 - step * pts, best_g0 + step * pts, 2 * pts + 1),
            np.linspace(best_g1 - step * pts, best_g1 + step * pts, 2 * pts + 1),
        )
        step /= 2.0

    return best_g0, best_g1


# =========================================================
# MAIN
# =========================================================

def main():
    print("\nLoading data...")
    df, x_cols = load_data()

    print("\nRunning single time-split experiment...")
    train, test = time_split_by_series(df)

    if train.empty or test.empty:
        raise ValueError("Train or test split is empty.")

    y_train = train[TARGET_COL].to_numpy(float)
    y_test = test[TARGET_COL].to_numpy(float)
    X_train = np.column_stack([np.ones(len(train)), train[x_cols].to_numpy(float)])
    X_test = np.column_stack([np.ones(len(test)), test[x_cols].to_numpy(float)])

    beta = fit_irls(X_train, y_train)
    p_train = sigmoid(X_train @ beta)
    p_test = sigmoid(X_test @ beta)

    if USE_NOISE_THRESHOLD_MODEL:
        noise_train = train["random_noise"].to_numpy(float)
        noise_test = test["random_noise"].to_numpy(float)
        g0, g1 = find_best_threshold(y_train, p_train, noise_train)
        yh_train = (p_train >= sigmoid(g0 + g1 * noise_train)).astype(int)
        yh_test = (p_test >= sigmoid(g0 + g1 * noise_test)).astype(int)
        threshold_summary = {
            "threshold_model": "sigmoid(gamma0 + gamma1 * standardized_random_noise)",
            "threshold_gamma0": g0,
            "threshold_gamma1": g1,
        }
    else:
        yh_train = (p_train >= FALLBACK_THRESHOLD).astype(int)
        yh_test = (p_test >= FALLBACK_THRESHOLD).astype(int)
        threshold_summary = {
            "threshold_model": "constant",
            "threshold_gamma0": None,
            "threshold_gamma1": None,
        }

    train_metrics = metrics(y_train, yh_train, p_train)
    test_metrics = metrics(y_test, yh_test, p_test)

    coef_names = ["const"] + x_cols
    coef_df = pd.DataFrame({
        "term": coef_names,
        "coef": beta,
        "odds_ratio": np.exp(beta),
    })

    print("\n" + "=" * 80)
    print("THRESHOLD MODEL")
    print("=" * 80)
    for k, v in threshold_summary.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 80)
    print("COEFFICIENTS")
    print("=" * 80)
    print(coef_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TRAIN METRICS")
    print("=" * 80)
    for k, v in train_metrics.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 80)
    print("TEST METRICS")
    print("=" * 80)
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # -----------------------------------------------------
    # New plotting section
    # -----------------------------------------------------
    plot_test_probability_distributions(y_test, p_test)

    out = {
        "train_n": train_metrics["n"],
        "train_positive_rate": train_metrics["positive_rate"],
        "train_predicted_positive_rate": train_metrics["predicted_positive_rate"],
        "train_auc": train_metrics["auc"],
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_specificity": train_metrics["specificity"],
        "train_f1": train_metrics["f1"],
        "train_tp": train_metrics["tp"],
        "train_tn": train_metrics["tn"],
        "train_fp": train_metrics["fp"],
        "train_fn": train_metrics["fn"],
        "test_n": test_metrics["n"],
        "test_positive_rate": test_metrics["positive_rate"],
        "test_predicted_positive_rate": test_metrics["predicted_positive_rate"],
        "test_auc": test_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_specificity": test_metrics["specificity"],
        "test_f1": test_metrics["f1"],
        "test_tp": test_metrics["tp"],
        "test_tn": test_metrics["tn"],
        "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"],
        "gap_f1": test_metrics["f1"] - train_metrics["f1"],
        "gap_auc": test_metrics["auc"] - train_metrics["auc"],
        "gap_accuracy": test_metrics["accuracy"] - train_metrics["accuracy"],
        "gap_precision": test_metrics["precision"] - train_metrics["precision"],
        "gap_recall": test_metrics["recall"] - train_metrics["recall"],
        "gap_specificity": test_metrics["specificity"] - train_metrics["specificity"],
        "gap_predicted_positive_rate": test_metrics["predicted_positive_rate"] - train_metrics["predicted_positive_rate"],
        "threshold_model": threshold_summary["threshold_model"],
        "threshold_gamma0": threshold_summary["threshold_gamma0"],
        "threshold_gamma1": threshold_summary["threshold_gamma1"],
    }

    pd.DataFrame([out]).to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV → {OUTPUT_CSV}")
    print(f"Saved plot → {OUTPUT_PLOT_ALL_TEST_PROBS}")
    print(f"Saved plot → {OUTPUT_PLOT_TEST_BY_CLASS}")
    print(f"Saved plot → {OUTPUT_PLOT_TEST_BOX}")

    print("\nNOTE ON AUC:")
    print("AUC = 0.38 is not 'good negative' by default.")
    print("It means the model ranks positives below negatives more often than not.")
    print("If that pattern is stable and intentional, then reversing the score would give about 1 - 0.38 = 0.62.")
    print("But as reported, 0.38 itself is worse than random, not better.")


if __name__ == "__main__":
    main()