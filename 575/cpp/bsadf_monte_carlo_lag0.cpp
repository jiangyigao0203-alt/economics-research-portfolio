/*
Zero-lag Monte Carlo calibration of endpoint-specific BSADF statistics.

Simulate Gaussian random walks, maximize the intercept-plus-lagged-level ADF
t-statistic over admissible start dates for each endpoint, and export an
empirical critical-value path. This standalone simulator uses 10,000 null paths
by default. It is distinct from the arbitrary-lag multi-market pipeline.
Output: part2_monte_carlo_cv_fast.csv in the process working directory.
*/

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <algorithm>
#include <limits>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::cout;
using std::endl;

// ============================================================
// SETTINGS
// Build instructions are in README.md. OpenMP is optional; without it the
// simulation loop runs serially. T and MIN_WINDOW count price observations.
// Defaults describe a large run; reduce them in a separate test configuration
// before a quick local check. Critical values apply only to zero-lag regressions.
// ============================================================
const int T = 6289;
const int MIN_WINDOW = 383;
const int MC_SIMS = 10000;
const double SIG_LEVEL = 0.95;
const int BASE_SEED = 12345;

const int PROGRESS_REFRESH_MS = 2000;
const long long REG_UPDATE_CHUNK = 200000;

// ============================================================
// UTILITIES
// ============================================================
std::string format_seconds(double seconds) {
    long long secs = static_cast<long long>(seconds);
    long long h = secs / 3600;
    long long m = (secs % 3600) / 60;
    long long s = secs % 60;

    char buffer[32];
    if (h > 0) {
        std::snprintf(buffer, sizeof(buffer), "%lld:%02lld:%02lld", h, m, s);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%02lld:%02lld", m, s);
    }
    return std::string(buffer);
}

std::string format_int(long long x) {
    std::string s = std::to_string(x);
    int n = static_cast<int>(s.size()) - 3;
    while (n > 0) {
        s.insert(n, ",");
        n -= 3;
    }
    return s;
}

std::string make_bar(double ratio, int width = 20) {
    ratio = std::max(0.0, std::min(1.0, ratio));
    int filled = static_cast<int>(ratio * width);

    std::string bar = "[";
    for (int i = 0; i < width; ++i) {
        bar += (i < filled ? "=" : "-");
    }
    bar += "]";
    return bar;
}

// ============================================================
// RANDOM WALK SIMULATION
// ============================================================
std::vector<double> simulate_random_walk(int size, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> y(size);
    double current = 0.0;
    for (int i = 0; i < size; ++i) {
        current += dist(gen);
        y[i] = current;
    }
    return y;
}

// ============================================================
// PREFIX SUMS
// x_i   = y_i
// d_i   = y_{i+1} - y_i
// obs i corresponds to regression row for time i+1
// Window [start, end] in y-space uses regression obs i=start..end-1
// ============================================================
struct PrefixSums {
    std::vector<double> px;    // sum x
    std::vector<double> px2;   // sum x^2
    std::vector<double> pd;    // sum d
    std::vector<double> pd2;   // sum d^2
    std::vector<double> pxd;   // sum x*d
};

// Precompute level/difference products so nested regression windows can
// update sufficient statistics without rebuilding an OLS design matrix.
PrefixSums build_prefix_sums(const std::vector<double>& y) {
    const int n = static_cast<int>(y.size());
    const int m = n - 1;

    PrefixSums pref;
    pref.px.assign(m + 1, 0.0);
    pref.px2.assign(m + 1, 0.0);
    pref.pd.assign(m + 1, 0.0);
    pref.pd2.assign(m + 1, 0.0);
    pref.pxd.assign(m + 1, 0.0);

    for (int i = 0; i < m; ++i) {
        double x = y[i];
        double d = y[i + 1] - y[i];

        pref.px[i + 1]  = pref.px[i]  + x;
        pref.px2[i + 1] = pref.px2[i] + x * x;
        pref.pd[i + 1]  = pref.pd[i]  + d;
        pref.pd2[i + 1] = pref.pd2[i] + d * d;
        pref.pxd[i + 1] = pref.pxd[i] + x * d;
    }

    return pref;
}

// ============================================================
// FAST ADF t-stat (no lags, with intercept)
// Regression:
//   d_i = alpha + beta * x_i + e_i
//
// Inputs are sufficient statistics over the current window.
// n = number of regression observations = end - start
// ============================================================
double adf_tstat_from_sums(
    int n,
    double sum_x,
    double sum_x2,
    double sum_d,
    double sum_d2,
    double sum_xd
) {
    if (n < 3) return std::numeric_limits<double>::quiet_NaN();

    double denom = n * sum_x2 - sum_x * sum_x;
    if (std::fabs(denom) < 1e-14) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double beta = (n * sum_xd - sum_x * sum_d) / denom;
    double alpha = (sum_d - beta * sum_x) / n;

    // SSE = sum (d - alpha - beta x)^2
    double sse =
        sum_d2
        - 2.0 * alpha * sum_d
        - 2.0 * beta * sum_xd
        + n * alpha * alpha
        + 2.0 * alpha * beta * sum_x
        + beta * beta * sum_x2;

    int dof = n - 2;
    if (dof <= 0) return std::numeric_limits<double>::quiet_NaN();

    // Guard against tiny negative values from floating-point noise
    if (sse < 0.0 && sse > -1e-10) sse = 0.0;
    if (sse < 0.0) return std::numeric_limits<double>::quiet_NaN();

    double s2 = sse / dof;

    // Var(beta) = s2 * n / denom
    double var_beta = s2 * n / denom;
    if (var_beta <= 0.0) return std::numeric_limits<double>::quiet_NaN();

    return beta / std::sqrt(var_beta);
}

// ============================================================
// MAIN
// ============================================================
int main() {
    auto global_start = std::chrono::high_resolution_clock::now();

    const int n_endpoints = T - MIN_WINDOW + 1;

    long long per_sim_regs = 0;
    for (int end = MIN_WINDOW - 1; end < T; ++end) {
        per_sim_regs += static_cast<long long>(end - MIN_WINDOW + 2);
    }
    long long global_total_regs = static_cast<long long>(MC_SIMS) * per_sim_regs;

    std::atomic<long long> global_sims_done{0};
    std::atomic<long long> global_regs_done{0};
    std::atomic<bool> work_finished{false};
    std::atomic<int> active_threads{0};

#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    bool openmp_enabled = true;
#else
    int max_threads = 1;
    bool openmp_enabled = false;
#endif

    cout << "============================================================\n";
    cout << "FAST MONTE CARLO BSADF (Recursive OLS stats)\n";
    cout << "============================================================\n";
    cout << "Sample size T            = " << T << "\n";
    cout << "Minimum window           = " << MIN_WINDOW << "\n";
    cout << "Monte Carlo simulations  = " << MC_SIMS << "\n";
    cout << "Endpoints                = " << n_endpoints << "\n";
    cout << "Total regressions        = " << format_int(global_total_regs) << "\n";
    cout << "OpenMP enabled           = " << (openmp_enabled ? "YES" : "NO") << "\n";
    cout << "Max OpenMP threads       = " << max_threads << "\n\n";

    // sim_mat[sim_idx][t]
    std::vector<std::vector<double>> sim_mat(
        MC_SIMS, std::vector<double>(T, std::numeric_limits<double>::quiet_NaN())
    );

    // ------------------------------------------------------------
    // Progress reporter
    // ------------------------------------------------------------
    std::thread progress_thread([&]() {
        while (!work_finished.load()) {
            long long sims = global_sims_done.load();
            long long regs = global_regs_done.load();
            int act = active_threads.load();

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - global_start).count();

            double sim_ratio = (MC_SIMS > 0) ? static_cast<double>(sims) / MC_SIMS : 0.0;
            double speed = (elapsed > 0.0) ? regs / elapsed : 0.0;
            double eta = (speed > 0.0) ? (global_total_regs - regs) / speed : 0.0;

            cout << "\r\x1b[2K"
                 << "Sims " << make_bar(sim_ratio, 20) << " " << sims << "/" << MC_SIMS
                 << " | Regs: " << format_int(regs) << "/" << format_int(global_total_regs)
                 << " | Cores Active: " << act
                 << " | Elapsed: " << format_seconds(elapsed)
                 << " | ETA: " << format_seconds(eta)
                 << std::flush;

            std::this_thread::sleep_for(std::chrono::milliseconds(PROGRESS_REFRESH_MS));
        }

        long long regs = global_regs_done.load();
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - global_start).count();

        cout << "\r\x1b[2K"
             << "Sims " << make_bar(1.0, 20) << " " << MC_SIMS << "/" << MC_SIMS
             << " | Regs: " << format_int(regs) << "/" << format_int(global_total_regs)
             << " | Done in: " << format_seconds(elapsed) << "\n";
    });

    // ------------------------------------------------------------
    // Parallel Monte Carlo
    // ------------------------------------------------------------
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int sim_idx = 0; sim_idx < MC_SIMS; ++sim_idx) {
        active_threads.fetch_add(1, std::memory_order_relaxed);

        std::vector<double> y = simulate_random_walk(T, BASE_SEED + sim_idx + 1);
        PrefixSums pref = build_prefix_sums(y);

        long long local_reg_counter = 0;

        // end is y-space endpoint
        for (int end = MIN_WINDOW - 1; end < T; ++end) {
            int start_max = end - MIN_WINDOW + 1;

            // Initialize with the smallest admissible window [start_max, end]
            int start0 = start_max;
            int n = end - start0; // regression observations = window length - 1

            double sum_x  = pref.px[end]  - pref.px[start0];
            double sum_x2 = pref.px2[end] - pref.px2[start0];
            double sum_d  = pref.pd[end]  - pref.pd[start0];
            double sum_d2 = pref.pd2[end] - pref.pd2[start0];
            double sum_xd = pref.pxd[end] - pref.pxd[start0];

            double best_stat = -std::numeric_limits<double>::infinity();

            // Expand window leftward: [start0, end], [start0-1, end], ..., [0, end]
            for (int start = start0; start >= 0; --start) {
                if (start < start0) {
                    // Add one observation i = start
                    double x_add = y[start];
                    double d_add = y[start + 1] - y[start];

                    sum_x  += x_add;
                    sum_x2 += x_add * x_add;
                    sum_d  += d_add;
                    sum_d2 += d_add * d_add;
                    sum_xd += x_add * d_add;
                    n += 1;
                }

                double stat = adf_tstat_from_sums(n, sum_x, sum_x2, sum_d, sum_d2, sum_xd);
                if (!std::isnan(stat) && stat > best_stat) {
                    best_stat = stat;
                }

                local_reg_counter++;
                if (local_reg_counter >= REG_UPDATE_CHUNK) {
                    global_regs_done.fetch_add(local_reg_counter, std::memory_order_relaxed);
                    local_reg_counter = 0;
                }
            }

            if (std::isfinite(best_stat)) {
                sim_mat[sim_idx][end] = best_stat;
            }
        }

        if (local_reg_counter > 0) {
            global_regs_done.fetch_add(local_reg_counter, std::memory_order_relaxed);
        }

        global_sims_done.fetch_add(1, std::memory_order_relaxed);
        active_threads.fetch_sub(1, std::memory_order_relaxed);
    }

    work_finished.store(true);
    progress_thread.join();

    // ------------------------------------------------------------
    // Aggregate critical values
    // ------------------------------------------------------------
    cout << "\nAggregating endpoint-specific critical values...\n";

    std::vector<double> critical_values(T, std::numeric_limits<double>::quiet_NaN());

    for (int t = 0; t < T; ++t) {
        std::vector<double> col_vals;
        col_vals.reserve(MC_SIMS);

        for (int sim = 0; sim < MC_SIMS; ++sim) {
            double v = sim_mat[sim][t];
            if (!std::isnan(v)) {
                col_vals.push_back(v);
            }
        }

        if (!col_vals.empty()) {
            std::sort(col_vals.begin(), col_vals.end());
            int rank = std::min(
                static_cast<int>(SIG_LEVEL * col_vals.size()),
                static_cast<int>(col_vals.size()) - 1
            );
            critical_values[t] = col_vals[rank];
        }
    }

    // ------------------------------------------------------------
    // Save CSV
    // ------------------------------------------------------------
    std::ofstream file("part2_monte_carlo_cv_fast.csv");
    file << "t_index,critical_value\n";
    for (int t = 0; t < T; ++t) {
        file << t << ",";
        if (std::isnan(critical_values[t])) file << "NaN\n";
        else file << critical_values[t] << "\n";
    }
    file.close();

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(total_end - global_start).count();

    cout << "\nDone. Saved: part2_monte_carlo_cv_fast.csv\n";
    cout << "Total elapsed time: " << format_seconds(total_elapsed) << "\n";

    return 0;
}