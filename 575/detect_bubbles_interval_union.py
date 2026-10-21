"""Run the zero-lag Bitcoin interval-union bubble-detection experiment.

Simulate random walks to calibrate the maximum ADF statistic across admissible
intervals, then mark the union of real-data intervals exceeding that threshold.
This procedure differs from endpoint-specific BSADF date-stamping. It expects
cryptos/btc-usd-max.csv, creates timestamped CSV/plot outputs, and can use all
CPU cores. Monte Carlo count, random seed, and plotting are configured below.
"""

import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = Path(__file__).resolve().parent / 'cryptos' / 'btc-usd-max.csv'
OUTPUT_DIR = Path(__file__).resolve().parent

RUN_TIMESTAMP = datetime.now(tz=UTC).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = OUTPUT_DIR / f"btc_naive_union_lag0_fast_{RUN_TIMESTAMP}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MC_OUTPUT_FILE = RUN_DIR / "btc_naive_union_mc_maxima.csv"
DETECTION_OUTPUT_FILE = RUN_DIR / "btc_naive_union_detection.csv"
PLOT_OUTPUT_FILE = RUN_DIR / "btc_naive_union_plot.png"

MIN_WINDOW_RULE = "psy"

MC_REPS = 200
MC_ALPHA = 0.95
RANDOM_SEED = 42

PRICE_IS_ALREADY_LOG = False
SHOW_PLOT = True

NUM_CORES = multiprocessing.cpu_count()
MC_TASKS_PER_CORE = 4
DETECTION_CHUNKS_PER_CORE = 4


# ============================================================
# UTILITIES
# ============================================================
def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def progress_bar(iteration, total, prefix="", suffix="", length=40):
    if total <= 0:
        total = 1
    percent = 100 * (iteration / total)
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:6.2f}% {suffix}", end="", flush=True)


def print_step(msg: str):
    print(msg, flush=True)


def overlay_bubble_segments(ax, x, y, is_bubble, linewidth=2.2):
    """Highlight contiguous bubble-labelled segments without joining separated episodes."""
    if len(y) < 2:
        return

    y = np.asarray(y, dtype=float)
    is_bubble = np.asarray(is_bubble, dtype=bool)

    valid = np.isfinite(y)
    bubble_seg = is_bubble[:-1] & is_bubble[1:] & valid[:-1] & valid[1:]
    start_idx = None
    n_seg = len(bubble_seg)

    for i in range(n_seg):
        if bubble_seg[i] and start_idx is None:
            start_idx = i
        elif (not bubble_seg[i]) and start_idx is not None:
            ax.plot(x[start_idx:i + 1], y[start_idx:i + 1], linewidth=linewidth, color="red")
            start_idx = None

    if start_idx is not None:
        ax.plot(x[start_idx:n_seg + 1], y[start_idx:n_seg + 1], linewidth=linewidth, color="red")


# ============================================================
# DATA PREP
# ============================================================
def prepare_daily_data(input_file: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)

    date_col = next(c for c in ["date", "caldt", "Date"] if c in df.columns)
    target_col = next(
        c for c in ["price", "close", "adj_close", "pe", "pe_ratio", "p_d", "target_metric"]
        if c in df.columns
    )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, target_col]].rename(columns={date_col: "date", target_col: "price"})
    df = df.dropna().sort_values("date").reset_index(drop=True)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).reset_index(drop=True)

    if PRICE_IS_ALREADY_LOG:
        df["log_price"] = df["price"].astype(float)
    else:
        df["log_price"] = np.log(df["price"].astype(float))

    return df


def get_min_window(T: int) -> int:
    """Convert the PSY sample-length rule into a minimum window measured in observations."""
    r0 = 0.01 + 1.8 / np.sqrt(T)
    return max(10, int(np.floor(r0 * T)))


# ============================================================
# PREFIX-SUM LAG-0 ADF CORE
# ============================================================
def build_prefix_sums_lag0(y: np.ndarray):
    """
    Lag-0 ADF regression:
        Δy_t = α + ρ y_{t-1} + ε_t

    For original series y[0],...,y[T-1], define regression rows indexed by j=0,...,T-2:
        x_j = y[j]
        z_j = y[j+1] - y[j]

    Prefix sums are built over x_j and z_j.
    """
    y = np.asarray(y, dtype=float)
    x = y[:-1]
    z = np.diff(y)

    px = np.concatenate(([0.0], np.cumsum(x)))
    pz = np.concatenate(([0.0], np.cumsum(z)))
    pxx = np.concatenate(([0.0], np.cumsum(x * x)))
    pxz = np.concatenate(([0.0], np.cumsum(x * z)))
    pzz = np.concatenate(([0.0], np.cumsum(z * z)))

    return x, z, px, pz, pxx, pxz, pzz


def interval_sum(prefix: np.ndarray, left: int, right: int) -> float:
    """
    Sum over prefix-backed array entries [left, right], inclusive.
    """
    if right < left:
        return 0.0
    return float(prefix[right + 1] - prefix[left])


def lag0_adf_tstat_from_prefix(
    px: np.ndarray,
    pz: np.ndarray,
    pxx: np.ndarray,
    pxz: np.ndarray,
    pzz: np.ndarray,
    start: int,
    end: int
) -> float:
    """
    Compute lag-0 ADF t-stat for interval y[start:end+1], inclusive in original y indexing.

    Regression rows correspond to j = start,...,end-1.
    So the number of regression observations is:
        n = end - start

    We estimate:
        z_j = α + ρ x_j + ε_j
    with x_j = y[j], z_j = Δy_{j+1}
    """
    left = start
    right = end - 1

    n = right - left + 1
    if n <= 2:
        return np.nan  # need positive dof since p=2

    sx = interval_sum(px, left, right)
    sz = interval_sum(pz, left, right)
    sxx = interval_sum(pxx, left, right)
    sxz = interval_sum(pxz, left, right)
    szz = interval_sum(pzz, left, right)

    # X'X and determinant
    # X'X = [[n, sx], [sx, sxx]]
    det = n * sxx - sx * sx
    if det <= 0 or not np.isfinite(det):
        return np.nan

    # OLS coefficient on x
    # beta1 = (n*sxz - sx*sz) / det
    beta1 = (n * sxz - sx * sz) / det
    beta0 = (sz - beta1 * sx) / n

    # RSS = sum z^2 - beta' X'z
    # X'z = [sz, sxz]
    rss = szz - (beta0 * sz + beta1 * sxz)
    dof = n - 2

    if dof <= 0:
        return np.nan

    # Numerical safeguard
    if rss < 0 and abs(rss) < 1e-10:
        rss = 0.0
    if rss < 0 or not np.isfinite(rss):
        return np.nan

    s2 = rss / dof

    # (X'X)^(-1) = (1/det) * [[sxx, -sx],[-sx,n]]
    # variance of beta1 = s2 * n / det
    var_beta1 = s2 * n / det

    if var_beta1 <= 0 or not np.isfinite(var_beta1):
        return np.nan

    tstat = beta1 / np.sqrt(var_beta1)
    return float(tstat)


def compute_max_interval_adf_lag0(y: np.ndarray, min_window: int) -> float:
    """
    Max lag-0 ADF over all valid intervals.
    O(T^2) total because each interval stat is O(1).
    """
    T = len(y)
    _, _, px, pz, pxx, pxz, pzz = build_prefix_sums_lag0(y)

    best_stat = -np.inf

    for end in range(min_window - 1, T):
        for start in range(end - min_window + 2):
            stat = lag0_adf_tstat_from_prefix(
                px=px, pz=pz, pxx=pxx, pxz=pxz, pzz=pzz,
                start=start, end=end
            )
            if np.isfinite(stat) and stat > best_stat:
                best_stat = stat

    return best_stat


# ============================================================
# MONTE CARLO WORKER
# ============================================================
def mc_worker_task(args):
    """
    One task handles n_reps independent null random walks.
    Each returns max lag-0 ADF over all intervals.
    """
    T, min_window, n_reps, seed = args
    rng = np.random.default_rng(seed)

    maxima = []

    for _ in range(n_reps):
        eps = rng.normal(loc=0.0, scale=1.0, size=T - 1)
        y = np.empty(T, dtype=float)
        y[0] = 0.0
        y[1:] = np.cumsum(eps)

        max_stat = compute_max_interval_adf_lag0(y, min_window=min_window)
        maxima.append(max_stat)

    return maxima


# ============================================================
# MONTE CARLO COORDINATOR
# ============================================================
def compute_mc_critical_value_multicore(
    T: int,
    min_window: int,
    n_reps: int,
    alpha: float,
    base_seed: int
):
    """
    Parallelize Monte Carlo over replications.
    """
    reps_per_task = max(1, math.ceil(n_reps / max(1, NUM_CORES * MC_TASKS_PER_CORE)))

    tasks = []
    reps_remaining = n_reps
    task_id = 0

    while reps_remaining > 0:
        batch = min(reps_per_task, reps_remaining)
        seed = base_seed + 100000 * task_id + 7
        tasks.append((T, min_window, batch, seed))
        reps_remaining -= batch
        task_id += 1

    print_step(f"Distributing {n_reps:,} Monte Carlo reps across {NUM_CORES} CPU cores...")

    start_time = time.time()
    reps_done = 0
    all_maxima = []

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(mc_worker_task, task) for task in tasks]

        for future in as_completed(futures):
            batch_maxima = future.result()
            all_maxima.extend(batch_maxima)

            reps_done += len(batch_maxima)

            elapsed = time.time() - start_time
            speed = reps_done / elapsed if elapsed > 0 else 0
            eta = (n_reps - reps_done) / speed if speed > 0 else 0

            suffix = f"Completed MC: {reps_done:,}/{n_reps:,} | ETA: {format_seconds(eta)}"
            progress_bar(reps_done, n_reps, prefix="MC Sim", suffix=suffix)

    print()

    maxima = np.asarray(all_maxima, dtype=float)
    maxima = maxima[np.isfinite(maxima)]

    if len(maxima) == 0:
        raise RuntimeError("Monte Carlo produced no finite maxima.")

    cv = float(np.quantile(maxima, alpha))
    return cv, maxima


# ============================================================
# REAL-DATA DETECTION WORKER
# ============================================================
def detection_worker_task(args):
    """
    For each assigned endpoint chunk:
      - compute endpoint max lag-0 ADF
      - mark any interval above cv in a difference array
    """
    y, min_window, cv, ends_chunk = args

    T = len(y)
    _, _, px, pz, pxx, pxz, pzz = build_prefix_sums_lag0(y)

    local_diff = np.zeros(T + 1, dtype=np.int64)
    endpoint_best_pairs = []

    local_best_stat = -np.inf
    interval_count = 0
    flagged_interval_count = 0

    for end in ends_chunk:
        endpoint_best = -np.inf

        for start in range(end - min_window + 2):
            stat = lag0_adf_tstat_from_prefix(
                px=px, pz=pz, pxx=pxx, pxz=pxz, pzz=pzz,
                start=start, end=end
            )
            interval_count += 1

            if np.isfinite(stat):
                endpoint_best = max(endpoint_best, stat)
                local_best_stat = max(local_best_stat, stat)

                if stat > cv:
                    local_diff[start] += 1
                    local_diff[end + 1] -= 1
                    flagged_interval_count += 1

        endpoint_best_pairs.append((end, endpoint_best))

    return (
        local_diff,
        endpoint_best_pairs,
        local_best_stat,
        interval_count,
        flagged_interval_count,
        len(ends_chunk)
    )


# ============================================================
# REAL-DATA DETECTION COORDINATOR
# ============================================================
def compute_union_detection_multicore(
    y: np.ndarray,
    min_window: int,
    cv: float
):
    """Mark observations belonging to any interval whose zero-lag ADF statistic exceeds the calibrated threshold."""
    T = len(y)
    all_endpoints = list(range(min_window - 1, T))
    n_endpoints = len(all_endpoints)

    chunk_size = max(1, n_endpoints // max(1, NUM_CORES * DETECTION_CHUNKS_PER_CORE))
    chunks = [all_endpoints[i:i + chunk_size] for i in range(0, n_endpoints, chunk_size)]
    tasks = [(y, min_window, cv, chunk) for chunk in chunks]

    print_step(f"Distributing {n_endpoints:,} endpoints across {NUM_CORES} CPU cores...")

    start_time = time.time()
    endpoints_done = 0

    global_diff = np.zeros(T + 1, dtype=np.int64)
    endpoint_max_adf = np.full(T, np.nan, dtype=float)

    global_best_stat = -np.inf
    total_interval_count = 0
    total_flagged_interval_count = 0

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(detection_worker_task, task) for task in tasks]

        for future in as_completed(futures):
            (
                local_diff,
                endpoint_best_pairs,
                local_best_stat,
                interval_count,
                flagged_interval_count,
                endpoints_in_chunk
            ) = future.result()

            global_diff += local_diff
            total_interval_count += interval_count
            total_flagged_interval_count += flagged_interval_count

            for end, best_stat in endpoint_best_pairs:
                if np.isfinite(best_stat):
                    endpoint_max_adf[end] = best_stat

            if np.isfinite(local_best_stat) and local_best_stat > global_best_stat:
                global_best_stat = local_best_stat

            endpoints_done += endpoints_in_chunk

            elapsed = time.time() - start_time
            speed = endpoints_done / elapsed if elapsed > 0 else 0
            eta = (n_endpoints - endpoints_done) / speed if speed > 0 else 0

            suffix = f"Done: {endpoints_done:,}/{n_endpoints:,} | ETA: {format_seconds(eta)}"
            progress_bar(endpoints_done, n_endpoints, prefix="Detect", suffix=suffix)

    print()

    bubble_mask = np.cumsum(global_diff[:-1]) > 0

    summary = {
        "max_interval_adf_real": float(global_best_stat) if np.isfinite(global_best_stat) else np.nan,
        "total_intervals_checked": int(total_interval_count),
        "flagged_intervals": int(total_flagged_interval_count),
        "flagged_share_of_points": float(np.mean(bubble_mask))
    }

    return bubble_mask, endpoint_max_adf, summary


# ============================================================
# PLOTTING
# ============================================================
def plot_detection_result(
    df: pd.DataFrame,
    output_file: Path,
    show_plot: bool = True
):
    """Plot the input series and interval-union labels, then save the configured figure."""
    x = df["date"].to_numpy()
    log_price = df["log_price"].to_numpy()
    endpoint_max_adf = df["endpoint_max_adf"].to_numpy()
    cv_series = df["mc_critical_value"].to_numpy()
    bubble = df["naive_union_bubble"].to_numpy(dtype=bool)

    _fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(14, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4, 1.0]}
    )

    ax1.plot(x, log_price, linewidth=1.2, label="Log Price")
    overlay_bubble_segments(ax1, x, log_price, bubble, linewidth=2.2)
    ax1.set_title("BTC | Log Price")
    ax1.set_ylabel("Log Price")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(x, endpoint_max_adf, linewidth=1.2, label="Endpoint Max ADF (lag 0)")
    ax2.plot(x, cv_series, linewidth=1.2, label="MC Critical Value")
    overlay_bubble_segments(ax2, x, endpoint_max_adf, bubble, linewidth=2.2)
    ax2.set_title("Endpoint Max ADF (lag 0) vs MC Critical Value")
    ax2.set_ylabel("Statistic")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(x, bubble.astype(int), linewidth=1.2, label="Naive Union Bubble Indicator")
    ax3.set_title("Naive Union Bubble Indicator")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("0 / 1")
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 90)
    print("BTC NAIVE UNION DETECTION | FAST LAG-0 PREFIX-SUM VERSION")
    print("=" * 90)
    print(f"Run directory             : {RUN_DIR}")
    print(f"MC output file            : {MC_OUTPUT_FILE}")
    print(f"Detection output file     : {DETECTION_OUTPUT_FILE}")
    print(f"Plot output file          : {PLOT_OUTPUT_FILE}")
    print("-" * 90)

    print_step("Loading data...")
    df = prepare_daily_data(INPUT_FILE)
    y = df["price"].to_numpy(dtype=float)

    T = len(y)
    min_window = get_min_window(T) if MIN_WINDOW_RULE.lower() == "psy" else int(MIN_WINDOW_RULE)

    print_step(f"Input file                = {INPUT_FILE}")
    print_step(f"Sample size               = {T:,}")
    print_step(f"Minimum window            = {min_window:,}")
    print_step("ADF lags                  = 0")
    print_step(f"MC reps                   = {MC_REPS:,}")
    print_step(f"MC alpha                  = {MC_ALPHA}")
    print_step(f"CPU cores                 = {NUM_CORES}")
    print_step("-" * 90)

    # --------------------------------------------------------
    # PART 1: MONTE CARLO
    # --------------------------------------------------------
    print()
    print("=" * 90)
    print("PART 1: MULTI-CORE MONTE CARLO CALIBRATION")
    print("=" * 90)

    mc_start = time.time()
    cv, mc_maxima = compute_mc_critical_value_multicore(
        T=T,
        min_window=min_window,
        n_reps=MC_REPS,
        alpha=MC_ALPHA,
        base_seed=RANDOM_SEED
    )
    mc_time = time.time() - mc_start

    mc_df = pd.DataFrame({
        "replication": np.arange(1, len(mc_maxima) + 1),
        "max_interval_adf": mc_maxima
    })
    mc_df.to_csv(MC_OUTPUT_FILE, index=False)

    print_step(f"Monte Carlo cutoff ({MC_ALPHA:.2%}) = {cv:.6f}")
    print_step(f"Saved MC maxima to       : {MC_OUTPUT_FILE}")
    print_step(f"MC elapsed               : {format_seconds(mc_time)}")
    print_step("-" * 90)

    # --------------------------------------------------------
    # PART 2: REAL DATA DETECTION
    # --------------------------------------------------------
    print()
    print("=" * 90)
    print("PART 2: MULTI-CORE REAL-DATA UNION DETECTION")
    print("=" * 90)

    det_start = time.time()
    bubble_mask, endpoint_max_adf, summary = compute_union_detection_multicore(
        y=y,
        min_window=min_window,
        cv=cv
    )
    det_time = time.time() - det_start

    df["endpoint_max_adf"] = endpoint_max_adf
    df["mc_critical_value"] = cv
    df["naive_union_bubble"] = bubble_mask.astype(int)

    df.to_csv(DETECTION_OUTPUT_FILE, index=False)

    print_step(f"Saved detection output to : {DETECTION_OUTPUT_FILE}")
    print_step(f"Detection elapsed         : {format_seconds(det_time)}")
    print_step(f"Max interval ADF (real)   = {summary['max_interval_adf_real']:.6f}")
    print_step(f"Total intervals checked   = {summary['total_intervals_checked']:,}")
    print_step(f"Flagged intervals         = {summary['flagged_intervals']:,}")
    print_step(f"Flagged share of points   = {summary['flagged_share_of_points']:.4%}")
    print_step("-" * 90)

    # --------------------------------------------------------
    # PART 3: PLOT
    # --------------------------------------------------------
    print()
    print("=" * 90)
    print("PART 3: PLOT")
    print("=" * 90)

    plot_detection_result(
        df=df,
        output_file=PLOT_OUTPUT_FILE,
        show_plot=SHOW_PLOT
    )

    print_step(f"Saved plot to             : {PLOT_OUTPUT_FILE}")

    total_time = mc_time + det_time
    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Total elapsed             : {format_seconds(total_time)}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()