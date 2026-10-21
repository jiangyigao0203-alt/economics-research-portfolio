"""
Multi-Seed Stability Runner
===========================
Edit the CONFIG block at the top, then just run:
    python multi_seed_runner.py
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# CONFIG  ← only edit this block
# =========================================================

INPUT_PANEL_FILE = Path(__file__).resolve().parent / 'dataset_build_outputs' / '111' / 'panel_dataset_full.csv'
OUTPUT_CSV       = Path(__file__).resolve().parent / 'mle_model_outputs' / 'multi_seed_results.csv'

N_RUNS    = 50       # how many seeds to run
TEST_SIZE = 0.30     # train/test split ratio

DATE_COL   = "date"
SERIES_COL = "series_name"
TARGET_COL = "is_bubble"

INCLUDE_VOL_LAG1            = True
INCLUDE_MARKET_TYPE_DUMMIES = True

USE_VOLATILITY_THRESHOLD_MODEL   = True
THRESHOLD_GRID_GAMMA0            = np.linspace(-4.0, 4.0, 81)
THRESHOLD_GRID_GAMMA1            = np.linspace(-4.0, 4.0, 81)
THRESHOLD_REFINE_ROUNDS          = 2
THRESHOLD_REFINE_STEP            = 0.5
THRESHOLD_REFINE_POINTS_PER_SIDE = 10

FALLBACK_THRESHOLD = 0.50
MAX_ITER           = 50
TOL                = 1e-8
RIDGE              = 1e-8


# =========================================================
# MATH HELPERS
# =========================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))

def normalize_target(s):
    if s.dtype == bool:
        return s.fillna(False).astype(int)
    return (
        s.astype(str).str.strip().str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0).astype(int)
    )

def compute_f1(y_true, y_hat):
    y, h  = np.asarray(y_true, int), np.asarray(y_hat, int)
    tp = int(((y == 1) & (h == 1)).sum())
    fp = int(((y == 0) & (h == 1)).sum())
    fn = int(((y == 1) & (h == 0)).sum())
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def compute_auc(y_true, p_pred):
    y, p  = np.asarray(y_true, int), np.asarray(p_pred, float)
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = pd.Series(p).rank(method="average").values[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# =========================================================
# DATA
# =========================================================

def load_data():
    df = pd.read_csv(INPUT_PANEL_FILE)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[DATE_COL, SERIES_COL, TARGET_COL]).copy()
    df[TARGET_COL] = normalize_target(df[TARGET_COL])

    x_cols = []
    if INCLUDE_VOL_LAG1:
        x_cols.append("garch_vol_lag1")
    if INCLUDE_MARKET_TYPE_DUMMIES:
        dummies = pd.get_dummies(df["market_type"], prefix="market_type", drop_first=True, dtype=float)
        df      = pd.concat([df, dummies], axis=1)
        x_cols.extend(dummies.columns.tolist())

    keep = [DATE_COL, SERIES_COL, "market_type", TARGET_COL, "garch_vol_lag1"] \
           + [c for c in x_cols if c != "garch_vol_lag1"]
    df = df[keep].dropna(subset=[TARGET_COL] + x_cols).reset_index(drop=True)

    print(f"Loaded {len(df)} rows | {df[SERIES_COL].nunique()} series | "
          f"positive rate = {df[TARGET_COL].mean():.4f}")
    return df, x_cols


def split_data(df, x_cols, seed):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for _, g in df.groupby("market_type", sort=True):
        series = sorted(g[SERIES_COL].dropna().astype(str).unique())
        n = len(series)
        if n == 0:
            continue
        rng.shuffle(series)
        n_test = 0 if n == 1 else max(1, min(int(np.floor(n * TEST_SIZE)), n - 1))
        train_parts.append(g[g[SERIES_COL].astype(str).isin(series[n_test:])].copy())
        if n_test > 0:
            test_parts.append(g[g[SERIES_COL].astype(str).isin(series[:n_test])].copy())

    def cat(parts):
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True).reset_index(drop=True)

    train, test = cat(train_parts), cat(test_parts)

    # standardize using train stats only
    cont = ["garch_vol_lag1"] if INCLUDE_VOL_LAG1 else []
    train, test = train.copy(), test.copy()
    for c in cont:
        mu = train[c].mean()
        sd = train[c].std() or 1.0
        train[c] = (train[c] - mu) / sd
        test[c]  = (test[c]  - mu) / sd

    return train, test


# =========================================================
# MODEL
# =========================================================

def fit_irls(X, y):
    k, beta = X.shape[1], np.zeros(X.shape[1])
    for _ in range(MAX_ITER):
        p    = np.clip(sigmoid(X @ beta), 1e-8, 1 - 1e-8)
        grad = X.T @ (y - p)
        H    = X.T @ (X * (p * (1 - p))[:, None]) + RIDGE * np.eye(k)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        beta += step
        if np.abs(step).max() < TOL:
            break
    return beta


def find_best_threshold(y, p, vol):
    best_f1, best_g0, best_g1 = -np.inf, 0.0, 0.0

    def search(g0s, g1s):
        nonlocal best_f1, best_g0, best_g1
        for g0 in g0s:
            for g1 in g1s:
                thr   = sigmoid(g0 + g1 * vol)
                y_hat = (p >= thr).astype(int)
                f1    = compute_f1(y, y_hat)
                if f1 > best_f1:
                    best_f1, best_g0, best_g1 = f1, float(g0), float(g1)

    search(THRESHOLD_GRID_GAMMA0, THRESHOLD_GRID_GAMMA1)

    step = THRESHOLD_REFINE_STEP
    pts  = THRESHOLD_REFINE_POINTS_PER_SIDE
    for _ in range(THRESHOLD_REFINE_ROUNDS):
        search(
            np.linspace(best_g0 - step * pts, best_g0 + step * pts, 2 * pts + 1),
            np.linspace(best_g1 - step * pts, best_g1 + step * pts, 2 * pts + 1),
        )
        step /= 2.0

    return best_g0, best_g1


# =========================================================
# ONE SEED
# =========================================================

def run_one_seed(df, x_cols, seed):
    try:
        train, test = split_data(df, x_cols, seed)
        if train.empty or test.empty:
            return None

        y_train = train[TARGET_COL].to_numpy(float)
        y_test  = test[TARGET_COL].to_numpy(float)
        X_train = np.column_stack([np.ones(len(train)), train[x_cols].to_numpy(float)])
        X_test  = np.column_stack([np.ones(len(test)),  test[x_cols].to_numpy(float)])

        beta    = fit_irls(X_train, y_train)
        p_train = sigmoid(X_train @ beta)
        p_test  = sigmoid(X_test  @ beta)

        if USE_VOLATILITY_THRESHOLD_MODEL:
            vol_train = train["garch_vol_lag1"].to_numpy(float)
            vol_test  = test["garch_vol_lag1"].to_numpy(float)
            g0, g1    = find_best_threshold(y_train, p_train, vol_train)
            yh_train  = (p_train >= sigmoid(g0 + g1 * vol_train)).astype(int)
            yh_test   = (p_test  >= sigmoid(g0 + g1 * vol_test)).astype(int)
        else:
            yh_train  = (p_train >= FALLBACK_THRESHOLD).astype(int)
            yh_test   = (p_test  >= FALLBACK_THRESHOLD).astype(int)

        def confusion(y_true, y_hat):
            y_true = np.asarray(y_true, int)
            y_hat = np.asarray(y_hat, int)
            tp = int(((y_true == 1) & (y_hat == 1)).sum())
            tn = int(((y_true == 0) & (y_hat == 0)).sum())
            fp = int(((y_true == 0) & (y_hat == 1)).sum())
            fn = int(((y_true == 1) & (y_hat == 0)).sum())
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

        train_metrics = metrics(y_train, yh_train, p_train)
        test_metrics = metrics(y_test, yh_test, p_test)

        result = {"seed": seed}
        for prefix, block in (("train", train_metrics), ("test", test_metrics)):
            for key, value in block.items():
                result[f"{prefix}_{key}"] = value

        result["gap_f1"] = result["test_f1"] - result["train_f1"]
        result["gap_auc"] = result["test_auc"] - result["train_auc"]
        result["gap_accuracy"] = result["test_accuracy"] - result["train_accuracy"]
        result["gap_precision"] = result["test_precision"] - result["train_precision"]
        result["gap_recall"] = result["test_recall"] - result["train_recall"]
        result["gap_specificity"] = result["test_specificity"] - result["train_specificity"]
        result["gap_predicted_positive_rate"] = result["test_predicted_positive_rate"] - result["train_predicted_positive_rate"]
        return result
    except Exception as e:
        logging.getLogger(__name__).exception("Operation failed in %s", __file__)
        print(f"  seed {seed} failed: {e}")
        return None


# =========================================================
# REPORT
# =========================================================

def report(df):
    def block(label, col):
        v = df[col].dropna().values
        print(f"\n  {label}")
        print(f"    mean={v.mean():.4f}  std={v.std():.4f}  "
              f"min={v.min():.4f}  p25={np.percentile(v,25):.4f}  "
              f"p50={np.percentile(v,50):.4f}  p75={np.percentile(v,75):.4f}  "
              f"max={v.max():.4f}")

    n = len(df)
    print("\n" + "="*65)
    print(f"  RESULTS  ({n} seeds)")
    print("="*65)

    print("\n--- F1 ---")
    block("Train F1",          "train_f1")
    block("Test  F1",          "test_f1")
    block("Gap  (test-train)", "gap_f1")
    print(f"\n  test_f1 > train_f1 in {(df.gap_f1 > 0).sum()} / {n} seeds "
          f"({100*(df.gap_f1 > 0).mean():.1f}%)")

    print("\n--- AUC ---")
    block("Train AUC",         "train_auc")
    block("Test  AUC",         "test_auc")
    block("Gap  (test-train)", "gap_auc")
    print(f"\n  test_auc > train_auc in {(df.gap_auc > 0).sum()} / {n} seeds "
          f"({100*(df.gap_auc > 0).mean():.1f}%)")

    print("\n--- Split check ---")
    block("Train positive rate", "train_pos_rate")
    block("Test  positive rate", "test_pos_rate")
    print()


# =========================================================
# MAIN
# =========================================================

print("\nLoading data...")
df, x_cols = load_data()

records = []
t_start = time.time()

print(f"\nRunning {N_RUNS} seeds...\n")

for seed in range(N_RUNS):
    t0      = time.time()
    result  = run_one_seed(df, x_cols, seed)
    elapsed = time.time() - t0

    if result:
        records.append(result)
        print(f"  seed={seed:3d} | "
              f"train_f1={result['train_f1']:.4f}  test_f1={result['test_f1']:.4f}  gap_f1={result['gap_f1']:+.4f} | "
              f"train_auc={result['train_auc']:.4f}  test_auc={result['test_auc']:.4f}  gap_auc={result['gap_auc']:+.4f} | "
              f"{elapsed:.1f}s")
    else:
        print(f"  seed={seed:3d} | skipped")

total = time.time() - t_start
print(f"\nDone. {len(records)}/{N_RUNS} seeds in {total:.1f}s")

if records:
    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")
    report(results_df)
