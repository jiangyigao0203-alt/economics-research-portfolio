import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = Path(__file__).resolve().parent / 'daily_pe.csv'
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "BSADF_pe_daily_1lag_multicore.csv"

ADF_LAGS = 2  
MIN_WINDOW_RULE = "psy"

# Number of CPU cores to use. Defaults to all available cores.
# Change to a specific number (e.g., 4) if you want to leave cores free for other tasks.
NUM_CORES = multiprocessing.cpu_count()

# ============================================================
# UTILITIES
# ============================================================
def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    m = (seconds % 3600) // 60
    s = seconds % 60
    h = seconds // 3600
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def progress_bar(iteration, total, prefix="", suffix="", length=40):
    if total <= 0: total = 1
    percent = 100 * (iteration / total)
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:6.2f}% {suffix}", end="", flush=True)

# ============================================================
# DATA PREP
# ============================================================
def prepare_daily_data(input_file: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    date_col = next(c for c in ["date", "caldt", "Date"] if c in df.columns)
    target_col = next(c for c in ["pe", "pe_ratio", "p_d", "close", "price", "target_metric"] if c in df.columns)
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, target_col]].rename(columns={date_col: "date", target_col: "target_metric"})
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df

def get_min_window(T: int) -> int:
    r0 = 0.01 + 1.8 / np.sqrt(T)
    return max(10, int(np.floor(r0 * T)))

# ============================================================
# WORKER FUNCTION (Must be at top-level for Multiprocessing)
# ============================================================
def worker_task(args):
    """
    This function runs independently on each CPU core.
    It receives the full 1D array, builds its own matrices, 
    and processes only its assigned chunk of endpoints.
    """
    y, lags, min_window, ends_chunk = args
    
    # 1. Build global matrices locally on this core (takes milliseconds)
    T = len(y)
    dy = np.diff(y)
    y_lag = y[:-1]
    
    Y_global = dy[lags:]
    X_cols = [np.ones(T - 1 - lags), y_lag[lags:]]
    for i in range(1, lags + 1):
        X_cols.append(dy[lags - i : len(dy) - i])
    
    X_global = np.column_stack(X_cols)
    k = X_global.shape[1]
    
    results = []
    
    # 2. Process the assigned chunk of endpoints
    for end in ends_chunk:
        best_stat = -np.inf
        start_max = end - min_window + 1
        end_idx = end - lags 

        for start in range(start_max, -1, -1):
            # Slice pre-computed matrices
            X_win = X_global[start:end_idx]
            Y_win = Y_global[start:end_idx]
            
            n = X_win.shape[0]
            dof = n - k
            
            if dof > 0:
                XtX = np.dot(X_win.T, X_win)
                try:
                    invXtX = np.linalg.inv(XtX)
                    XtY = np.dot(X_win.T, Y_win)
                    beta_hat = np.dot(invXtX, XtY)
                    
                    resid = Y_win - np.dot(X_win, beta_hat)
                    s2 = np.dot(resid, resid) / dof
                    var_beta = s2 * invXtX[1, 1] 
                    
                    if var_beta > 0:
                        stat = beta_hat[1] / np.sqrt(var_beta)
                        best_stat = max(best_stat, stat)
                except np.linalg.LinAlgError:
                    pass 
        
        # Save the best stat for this specific endpoint
        results.append((end, best_stat))
        
    return results

# ============================================================
# MULTICORE COORDINATOR
# ============================================================
def compute_bsadf_multicore(y: np.ndarray, min_window: int, lags: int) -> np.ndarray:
    T = len(y)
    bsadf = np.full(T, np.nan)
    
    # Create a list of all endpoints we need to calculate
    all_endpoints = list(range(min_window - 1, T))
    n_endpoints = len(all_endpoints)
    
    # Break the endpoints into chunks for the CPU cores
    # Multiplying by 4 ensures cores that finish early can pick up more work
    chunk_size = max(1, n_endpoints // (NUM_CORES * 4))
    chunks = [all_endpoints[i:i + chunk_size] for i in range(0, n_endpoints, chunk_size)]
    
    # Package the arguments for the workers
    tasks = [(y, lags, min_window, chunk) for chunk in chunks]
    
    print(f"Distributing {n_endpoints:,} endpoints across {NUM_CORES} CPU cores...")
    
    start_time = time.time()
    endpoints_done = 0

    # Launch the multi-core process pool
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        # Map futures to their tasks
        futures = [executor.submit(worker_task, task) for task in tasks]
        
        # Process results as they finish (unordered)
        for future in as_completed(futures):
            result_chunk = future.result()
            
            # Unpack the results into the main BSADF array
            for end, stat in result_chunk:
                if np.isfinite(stat):
                    bsadf[end] = stat
                    
            endpoints_done += len(result_chunk)
            
            # Update progress bar
            elapsed = time.time() - start_time
            speed = endpoints_done / elapsed if elapsed > 0 else 0
            eta = (n_endpoints - endpoints_done) / speed if speed > 0 else 0
            
            suffix = f"Done: {endpoints_done:,}/{n_endpoints:,} | ETA: {format_seconds(eta)}"
            progress_bar(endpoints_done, n_endpoints, prefix="Computing", suffix=suffix)

    print() # Print new line after progress bar finishes
    return bsadf

# ============================================================
# MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"PART 1: MULTI-CORE BSADF (Using {NUM_CORES} Cores)")
    print("=" * 70)

    print("Loading data...")
    df = prepare_daily_data(INPUT_FILE)
    y = df["target_metric"].to_numpy()
    
    T = len(y)
    min_window = get_min_window(T)

    print(f"Sample size (Days)        = {T:,}")
    print(f"Minimum window (Days)     = {min_window:,}")
    print(f"ADF Lags Applied          = {ADF_LAGS}")
    print("-" * 70)
    
    # Run the multicore calculation
    total_start = time.time()
    df["bsadf"] = compute_bsadf_multicore(y, min_window, lags=ADF_LAGS)
    total_time = time.time() - total_start

    # Clean up column names for final CSV
    df = df.rename(columns={"target_metric": "price"})
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nDONE. Saved highly accurate output to: {OUTPUT_FILE}")
    print(f"Total processing time: {format_seconds(total_time)}")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support() 
    main()