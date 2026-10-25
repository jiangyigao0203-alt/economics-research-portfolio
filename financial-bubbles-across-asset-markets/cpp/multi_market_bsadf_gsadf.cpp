/*
Calibrated multi-market BSADF and GSADF analysis.

For each market and configured ADF lag, simulate random-walk null series,
estimate endpoint-specific BSADF and whole-sample GSADF critical values, and
export dated bubble labels plus a GSADF summary. Short positive runs are
filtered using MIN_BUBBLE_DURATION. This filter operates retrospectively.
*/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

// ============================================================
// USER SETTINGS
// ============================================================
// Build with CMake as described in README.md; execute from financial-bubbles-across-asset-markets/cpp.
// Input subdirectories are ETFS, ExchangeRate, and crypto.
// CSV columns 1 and 2 are date and positive price; later columns are ignored.
// Prices are logged before estimation. Input order is chronological unless
// SORT_BY_DATE_LEXICOGRAPHIC is enabled for consistently sortable date strings.
// Critical values are cached by sample size, minimum window, lag, simulation
// count, and quantile. ADF_LAG_LIST controls the lag specifications tested.

const string DATASET_ROOT_DIR = "../data/markets";
const string OUTPUT_PARENT_DIR = "../outputs/bsadf_gsadf";

const vector<int> ADF_LAG_LIST  = {2, 3};   // Lagged differences in each ADF regression.
const int MONTE_CARLO_SIMS      = 500;      // Null paths per calibration.
const double SIG_LEVEL          = 0.95;     // 95% critical value
const int MIN_BUBBLE_DURATION   = 5;
const bool SORT_BY_DATE_LEXICOGRAPHIC = false;
const int PROGRESS_REFRESH_MS   = 1000;

// ============================================================
// DATA STRUCTURES
// ============================================================
struct MarketRow {
    string date;
    double raw_price = numeric_limits<double>::quiet_NaN();
    double log_price = numeric_limits<double>::quiet_NaN();

    double bsadf = numeric_limits<double>::quiet_NaN();
    double critical_value = numeric_limits<double>::quiet_NaN();
};

struct MarketData {
    string market_name;
    string market_type;   // ETF / ExchangeRate / crypto
    string input_path;
    vector<MarketRow> rows;
};

struct ProgressState {
    atomic<long long> done{0};
    long long total = 1;
    atomic<bool> finished{false};
    string label;
};

struct GsadfSummaryRow {
    string market_name;
    string market_type;
    string input_file;
    int lag = 0;
    int observations = 0;
    int min_window = 0;
    string first_date;
    string last_date;
    double gsadf = numeric_limits<double>::quiet_NaN();
    double gsadf_critical_value = numeric_limits<double>::quiet_NaN();
    bool gsadf_reject = false;
};

struct MCKey {
    int T = 0;
    int min_window = 0;
    int lag = 0;
    int sims = 0;
    double sig_level = 0.0;
    bool operator<(const MCKey& other) const {
        if (T != other.T) return T < other.T;
        if (min_window != other.min_window) return min_window < other.min_window;
        if (lag != other.lag) return lag < other.lag;
        if (sims != other.sims) return sims < other.sims;
        return sig_level < other.sig_level;
    }
};

// ============================================================
// UTILITIES
// ============================================================
string trim(const string& s) {
    size_t a = 0;
    while (a < s.size() && isspace(static_cast<unsigned char>(s[a]))) a++;
    size_t b = s.size();
    while (b > a && isspace(static_cast<unsigned char>(s[b - 1]))) b--;
    return s.substr(a, b - a);
}

string to_lower_copy(string s) {
    for (char& c : s) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    return s;
}

string sanitize_filename(string s) {
    for (char& c : s) {
        if (!(isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.')) {
            c = '_';
        }
    }
    while (!s.empty() && (s.back() == '.' || s.back() == ' ')) s.pop_back();
    if (s.empty()) s = "market";
    return s;
}

string now_timestamp_for_folder() {
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &t);
#else
    localtime_r(&t, &local_tm);
#endif
    ostringstream oss;
    oss << put_time(&local_tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

bool safe_stod(const string& s, double& out) {
    string x = trim(s);
    if (x.empty()) return false;
    try {
        size_t pos = 0;
        out = stod(x, &pos);
        if (pos != x.size()) return false;
        return true;
    } catch (...) {
        return false;
    }
}

vector<string> split_csv_line(const string& line) {
    vector<string> out;
    string cur;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

string bool_to_text(bool x) {
    return x ? "True" : "False";
}

string format_seconds(double seconds) {
    if (seconds < 0) seconds = 0;
    long long sec = static_cast<long long>(seconds);
    long long h = sec / 3600;
    long long m = (sec % 3600) / 60;
    long long s = sec % 60;

    ostringstream oss;
    if (h > 0) {
        oss << h << ":" << setw(2) << setfill('0') << m
            << ":" << setw(2) << setfill('0') << s;
    } else {
        oss << setw(2) << setfill('0') << m
            << ":" << setw(2) << setfill('0') << s;
    }
    return oss.str();
}

void print_bar_line(const string& label, long long done, long long total, double elapsed_sec) {
    if (total <= 0) total = 1;
    double ratio = max(0.0, min(1.0, static_cast<double>(done) / static_cast<double>(total)));
    int width = 36;
    int filled = static_cast<int>(ratio * width);

    string bar(filled, char(219));
    bar += string(width - filled, '-');

    double speed = elapsed_sec > 0 ? done / elapsed_sec : 0.0;
    double eta = speed > 0 ? (total - done) / speed : 0.0;

    cout << "\r" << left << setw(28) << label
         << " |" << bar << "| "
         << fixed << setprecision(2) << setw(6) << (100.0 * ratio) << "% "
         << "done " << done << "/" << total
         << " | elapsed " << format_seconds(elapsed_sec)
         << " | ETA " << format_seconds(eta) << flush;
}

void progress_thread_runner(ProgressState* ps) {
    auto t0 = chrono::steady_clock::now();
    while (!ps->finished.load()) {
        auto t1 = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(t1 - t0).count();
        print_bar_line(ps->label, ps->done.load(), ps->total, elapsed);
        this_thread::sleep_for(chrono::milliseconds(PROGRESS_REFRESH_MS));
    }
    auto t1 = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(t1 - t0).count();
    print_bar_line(ps->label, ps->total, ps->total, elapsed);
    cout << "\n";
}

int get_min_window(int T) {
    double r0 = 0.01 + 1.8 / sqrt(static_cast<double>(T));
    return max(10, static_cast<int>(floor(r0 * T)));
}

// ============================================================
// MATRIX HELPERS
// ============================================================
bool invert_matrix(vector<vector<double>> A, vector<vector<double>>& invA) {
    int n = static_cast<int>(A.size());
    invA.assign(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) invA[i][i] = 1.0;

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int r = col + 1; r < n; ++r) {
            if (fabs(A[r][col]) > fabs(A[pivot][col])) pivot = r;
        }
        if (fabs(A[pivot][col]) < 1e-14) return false;

        if (pivot != col) {
            swap(A[pivot], A[col]);
            swap(invA[pivot], invA[col]);
        }

        double div = A[col][col];
        for (int j = 0; j < n; ++j) {
            A[col][j] /= div;
            invA[col][j] /= div;
        }

        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            double factor = A[r][col];
            for (int j = 0; j < n; ++j) {
                A[r][j] -= factor * A[col][j];
                invA[r][j] -= factor * invA[col][j];
            }
        }
    }
    return true;
}

// ============================================================
// FILE IO
// ============================================================
vector<fs::path> list_csv_files(const fs::path& folder) {
    vector<fs::path> files;
    if (!fs::exists(folder)) return files;

    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        string ext = to_lower_copy(entry.path().extension().string());
        if (ext == ".csv") files.push_back(entry.path());
    }
    sort(files.begin(), files.end());
    return files;
}

string folder_to_market_type(const string& folder_name) {
    if (folder_name == "ETFS") return "ETF";
    if (folder_name == "ExchangeRate") return "ExchangeRate";
    if (folder_name == "crypto") return "crypto";
    return folder_name;
}

MarketData read_market_csv_first_two_columns(const fs::path& path, const string& market_type) {
    ifstream fin(path);
    if (!fin.is_open()) {
        throw runtime_error("Could not open file: " + path.string());
    }

    string header;
    if (!getline(fin, header)) {
        throw runtime_error("Empty file: " + path.string());
    }

    MarketData md;
    md.market_name = path.stem().string();
    md.market_type = market_type;
    md.input_path = path.string();

    string line;
    while (getline(fin, line)) {
        if (trim(line).empty()) continue;

        vector<string> parts = split_csv_line(line);
        if (parts.size() < 2) continue;

        string date_str = trim(parts[0]);
        string price_str = trim(parts[1]);

        double raw_price = numeric_limits<double>::quiet_NaN();
        if (date_str.empty()) continue;
        if (!safe_stod(price_str, raw_price)) continue;
        if (!(raw_price > 0.0)) continue;

        MarketRow row;
        row.date = date_str;
        row.raw_price = raw_price;
        row.log_price = log(raw_price);
        md.rows.push_back(row);
    }

    if (md.rows.empty()) {
        throw runtime_error("No valid data rows found in file: " + path.string());
    }

    if (SORT_BY_DATE_LEXICOGRAPHIC) {
        sort(md.rows.begin(), md.rows.end(),
             [](const MarketRow& a, const MarketRow& b) { return a.date < b.date; });
    }

    return md;
}

// ============================================================
// GENERIC ADF t-STAT FOR A GIVEN WINDOW AND LAG
// Model:
//   Δy_t = alpha + beta * y_{t-1} + sum_{j=1}^p phi_j * Δy_{t-j} + e_t
// Test stat = t-stat of beta
//
// Window is inclusive in y-space: [start, end]
// ============================================================
// Fit an intercept, lagged log level, and lagged differences on inclusive
// price indices [start, end]; return the right-tailed t-statistic on the level.
bool adf_tstat_window_general(
    const vector<double>& y,
    int start,
    int end,
    int lag,
    double& tstat_out
) {
    if (lag < 0) return false;

    int first_t = start + lag + 1;
    int last_t  = end;

    int n = last_t - first_t + 1;
    int k = 2 + lag; // const + y_{t-1} + lagged diffs

    if (n <= k) return false;

    vector<vector<double>> XtX(k, vector<double>(k, 0.0));
    vector<double> XtY(k, 0.0);
    double YY = 0.0;

    for (int t = first_t; t <= last_t; ++t) {
        vector<double> x(k, 0.0);
        x[0] = 1.0;
        x[1] = y[t - 1];

        for (int j = 1; j <= lag; ++j) {
            x[1 + j] = y[t - j] - y[t - j - 1];
        }

        double dep = y[t] - y[t - 1];

        for (int a = 0; a < k; ++a) {
            XtY[a] += x[a] * dep;
            for (int b = 0; b < k; ++b) {
                XtX[a][b] += x[a] * x[b];
            }
        }
        YY += dep * dep;
    }

    vector<vector<double>> invXtX;
    if (!invert_matrix(XtX, invXtX)) return false;

    vector<double> beta_hat(k, 0.0);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            beta_hat[i] += invXtX[i][j] * XtY[j];
        }
    }

    double betaXtY = 0.0;
    for (int i = 0; i < k; ++i) betaXtY += beta_hat[i] * XtY[i];

    double sse = YY - betaXtY;
    double dof = static_cast<double>(n - k);

    if (dof <= 0.0) return false;
    if (sse < 0.0 && sse > -1e-10) sse = 0.0;
    if (sse < 0.0) return false;

    double s2 = sse / dof;
    double var_beta = s2 * invXtX[1][1];

    if (!(var_beta > 0.0)) return false;

    tstat_out = beta_hat[1] / sqrt(var_beta);
    return isfinite(tstat_out);
}

// ============================================================
// BSADF
// ============================================================
vector<double> compute_bsadf_general(
    const vector<double>& y,
    int min_window,
    int lag,
    const string& progress_label
) {
    int T = static_cast<int>(y.size());
    vector<double> bsadf(T, numeric_limits<double>::quiet_NaN());

    if (T < 6) return bsadf;
    if (min_window < lag + 5) min_window = lag + 5;

    vector<int> endpoints;
    for (int e = min_window - 1; e < T; ++e) endpoints.push_back(e);

    ProgressState ps;
    ps.total = static_cast<long long>(endpoints.size());
    ps.label = progress_label;

    thread pth(progress_thread_runner, &ps);

    unsigned int n_threads = thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;

    atomic<size_t> next_idx(0);
    vector<thread> workers;

    auto worker = [&]() {
        while (true) {
            size_t idx = next_idx.fetch_add(1);
            if (idx >= endpoints.size()) break;

            int e = endpoints[idx];
            double best = -numeric_limits<double>::infinity();

            int start_max = e - min_window + 1;
            for (int start = start_max; start >= 0; --start) {
                double tstat = numeric_limits<double>::quiet_NaN();
                if (adf_tstat_window_general(y, start, e, lag, tstat)) {
                    if (tstat > best) best = tstat;
                }
            }

            if (isfinite(best)) bsadf[e] = best;
            ps.done.fetch_add(1);
        }
    };

    for (unsigned int i = 0; i < n_threads; ++i) workers.emplace_back(worker);
    for (auto& th : workers) th.join();

    ps.finished.store(true);
    pth.join();

    return bsadf;
}

// ============================================================
// GSADF
// ============================================================
double compute_gsadf_general(
    const vector<double>& y,
    int min_window,
    int lag,
    const string& progress_label
) {
    int T = static_cast<int>(y.size());
    if (T < 6) return numeric_limits<double>::quiet_NaN();
    if (min_window < lag + 5) min_window = lag + 5;

    vector<int> endpoints;
    for (int e = min_window - 1; e < T; ++e) endpoints.push_back(e);

    ProgressState ps;
    ps.total = static_cast<long long>(endpoints.size());
    ps.label = progress_label;

    thread pth(progress_thread_runner, &ps);

    unsigned int n_threads = thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;

    atomic<size_t> next_idx(0);
    vector<thread> workers;

    double global_best = -numeric_limits<double>::infinity();
    mutex mtx;

    auto worker = [&]() {
        double local_best = -numeric_limits<double>::infinity();

        while (true) {
            size_t idx = next_idx.fetch_add(1);
            if (idx >= endpoints.size()) break;

            int e = endpoints[idx];
            int start_max = e - min_window + 1;

            for (int start = start_max; start >= 0; --start) {
                double tstat = numeric_limits<double>::quiet_NaN();
                if (adf_tstat_window_general(y, start, e, lag, tstat)) {
                    if (tstat > local_best) local_best = tstat;
                }
            }

            ps.done.fetch_add(1);
        }

        lock_guard<mutex> lock(mtx);
        if (local_best > global_best) global_best = local_best;
    };

    for (unsigned int i = 0; i < n_threads; ++i) workers.emplace_back(worker);
    for (auto& th : workers) th.join();

    ps.finished.store(true);
    pth.join();

    if (!isfinite(global_best)) return numeric_limits<double>::quiet_NaN();
    return global_best;
}

// ============================================================
// MONTE CARLO
// ============================================================
vector<double> simulate_random_walk(int T, uint64_t seed) {
    mt19937_64 gen(seed);
    normal_distribution<double> dist(0.0, 1.0);

    vector<double> y(T, 0.0);
    double cur = 0.0;
    for (int i = 0; i < T; ++i) {
        cur += dist(gen);
        y[i] = cur;
    }
    return y;
}

// Calibrate a separate null quantile for each admissible endpoint. Each
// simulation writes its own row, so worker scheduling does not change its seed.
vector<double> compute_monte_carlo_bsadf_critical_values(
    int T,
    int min_window,
    int lag,
    int sims,
    double sig_level,
    const string& progress_label
) {
    vector<double> cv(T, numeric_limits<double>::quiet_NaN());
    if (T < 6) return cv;

    int first_valid = min_window - 1;
    int n_endpoints = T - first_valid;
    if (n_endpoints <= 0) return cv;

    vector<vector<float>> sim_stats(sims, vector<float>(n_endpoints, numeric_limits<float>::quiet_NaN()));

    ProgressState ps;
    ps.total = sims;
    ps.label = progress_label;
    thread pth(progress_thread_runner, &ps);

    unsigned int n_threads = thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;

    atomic<int> next_sim(0);
    vector<thread> workers;

    auto worker = [&]() {
        while (true) {
            int sim = next_sim.fetch_add(1);
            if (sim >= sims) break;

            uint64_t seed = 123456789ULL
                          + static_cast<uint64_t>(sim) * 7919ULL
                          + static_cast<uint64_t>(T) * 104729ULL
                          + static_cast<uint64_t>(lag) * 1009ULL;

            vector<double> y = simulate_random_walk(T, seed);

            for (int e = first_valid; e < T; ++e) {
                double best = -numeric_limits<double>::infinity();
                int start_max = e - min_window + 1;

                for (int start = start_max; start >= 0; --start) {
                    double tstat = numeric_limits<double>::quiet_NaN();
                    if (adf_tstat_window_general(y, start, e, lag, tstat)) {
                        if (tstat > best) best = tstat;
                    }
                }

                if (isfinite(best)) {
                    sim_stats[sim][e - first_valid] = static_cast<float>(best);
                }
            }

            ps.done.fetch_add(1);
        }
    };

    for (unsigned int i = 0; i < n_threads; ++i) workers.emplace_back(worker);
    for (auto& th : workers) th.join();

    ps.finished.store(true);
    pth.join();

    cout << "Aggregating BSADF Monte Carlo critical values...\n";
    for (int off = 0; off < n_endpoints; ++off) {
        vector<float> vals;
        vals.reserve(sims);

        for (int s = 0; s < sims; ++s) {
            float v = sim_stats[s][off];
            if (!isnan(v)) vals.push_back(v);
        }
        if (vals.empty()) continue;

        sort(vals.begin(), vals.end());
        int rank = static_cast<int>(floor(sig_level * static_cast<double>(vals.size())));
        if (rank >= static_cast<int>(vals.size())) rank = static_cast<int>(vals.size()) - 1;
        if (rank < 0) rank = 0;

        cv[first_valid + off] = static_cast<double>(vals[rank]);
    }

    return cv;
}

// Calibrate one quantile of the maximum statistic across all admissible
// starts and endpoints; this is a whole-sample threshold, not a dated path.
double compute_monte_carlo_gsadf_critical_value(
    int T,
    int min_window,
    int lag,
    int sims,
    double sig_level,
    const string& progress_label
) {
    if (T < 6) return numeric_limits<double>::quiet_NaN();

    vector<float> sim_stats(sims, numeric_limits<float>::quiet_NaN());

    ProgressState ps;
    ps.total = sims;
    ps.label = progress_label;
    thread pth(progress_thread_runner, &ps);

    unsigned int n_threads = thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;

    atomic<int> next_sim(0);
    vector<thread> workers;

    auto worker = [&]() {
        while (true) {
            int sim = next_sim.fetch_add(1);
            if (sim >= sims) break;

            uint64_t seed = 987654321ULL
                          + static_cast<uint64_t>(sim) * 7919ULL
                          + static_cast<uint64_t>(T) * 104729ULL
                          + static_cast<uint64_t>(lag) * 1009ULL;

            vector<double> y = simulate_random_walk(T, seed);

            double gsadf_best = -numeric_limits<double>::infinity();
            for (int e = min_window - 1; e < T; ++e) {
                int start_max = e - min_window + 1;
                for (int start = start_max; start >= 0; --start) {
                    double tstat = numeric_limits<double>::quiet_NaN();
                    if (adf_tstat_window_general(y, start, e, lag, tstat)) {
                        if (tstat > gsadf_best) gsadf_best = tstat;
                    }
                }
            }

            if (isfinite(gsadf_best)) {
                sim_stats[sim] = static_cast<float>(gsadf_best);
            }

            ps.done.fetch_add(1);
        }
    };

    for (unsigned int i = 0; i < n_threads; ++i) workers.emplace_back(worker);
    for (auto& th : workers) th.join();

    ps.finished.store(true);
    pth.join();

    vector<float> vals;
    vals.reserve(sims);
    for (float v : sim_stats) {
        if (!isnan(v)) vals.push_back(v);
    }
    if (vals.empty()) return numeric_limits<double>::quiet_NaN();

    sort(vals.begin(), vals.end());
    int rank = static_cast<int>(floor(sig_level * static_cast<double>(vals.size())));
    if (rank >= static_cast<int>(vals.size())) rank = static_cast<int>(vals.size()) - 1;
    if (rank < 0) rank = 0;

    return static_cast<double>(vals[rank]);
}

// ============================================================
// BUBBLE FILTER
// ============================================================
// Remove positive runs shorter than min_duration. A run is evaluated using
// its eventual length, so this filter uses future observations within that run.
vector<bool> apply_min_duration_filter(const vector<bool>& raw_flag, int min_duration) {
    vector<bool> out(raw_flag.size(), false);
    if (min_duration <= 1) return raw_flag;

    int n = static_cast<int>(raw_flag.size());
    int i = 0;
    while (i < n) {
        if (!raw_flag[i]) {
            ++i;
            continue;
        }
        int j = i;
        while (j < n && raw_flag[j]) ++j;
        int len = j - i;
        if (len >= min_duration) {
            for (int k = i; k < j; ++k) out[k] = true;
        }
        i = j;
    }
    return out;
}

// ============================================================
// OUTPUT WRITERS
// ============================================================
// Export raw and logged prices separately, then rename a completed temporary
// file. The Python analyses consume logged price and filtered is_bubble.
void write_bsadf_result_csv_atomic(
    const string& output_dir,
    const MarketData& md,
    const vector<MarketRow>& rows,
    int lag,
    int min_bubble_duration
) {
    string safe_market = sanitize_filename(md.market_name);
    string safe_type   = sanitize_filename(md.market_type);

    string filename = "BSADF_" + safe_market + "_" + safe_type + "_lag" + to_string(lag) + ".csv";

    fs::path final_path = fs::path(output_dir) / filename;
    fs::path temp_path  = fs::path(output_dir) / (filename + ".tmp");

    vector<bool> raw_bubble(rows.size(), false);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (isfinite(rows[i].bsadf) && isfinite(rows[i].critical_value)) {
            raw_bubble[i] = (rows[i].bsadf > rows[i].critical_value);
        }
    }
    vector<bool> filtered_bubble = apply_min_duration_filter(raw_bubble, min_bubble_duration);

    ofstream fout(temp_path);
    if (!fout.is_open()) {
        throw runtime_error("Could not write temp BSADF output: " + temp_path.string());
    }

    fout << "market_name,market_type,input_file,lag,date,target_metric,price,bsadf,row_index_bsadf,critical_value,row_index_cv,bsadf_notna,cv_notna,can_compare,is_bubble_raw,is_bubble\n";
    fout << setprecision(15);

    for (size_t i = 0; i < rows.size(); ++i) {
        bool bsadf_notna = isfinite(rows[i].bsadf);
        bool cv_notna    = isfinite(rows[i].critical_value);
        bool can_compare = bsadf_notna && cv_notna;

        fout << md.market_name << ","
             << md.market_type << ","
             << fs::path(md.input_path).filename().string() << ","
             << lag << ","
             << rows[i].date << ","
             << rows[i].raw_price << ","
             << rows[i].log_price << ",";

        if (bsadf_notna) fout << rows[i].bsadf;
        fout << ",";

        fout << i << ",";

        if (cv_notna) fout << rows[i].critical_value;
        fout << ",";

        if (cv_notna) fout << i;
        fout << ",";

        fout << bool_to_text(bsadf_notna) << ","
             << bool_to_text(cv_notna) << ","
             << bool_to_text(can_compare) << ","
             << bool_to_text(raw_bubble[i]) << ","
             << bool_to_text(filtered_bubble[i]) << "\n";
    }

    fout.close();
    if (fs::exists(final_path)) fs::remove(final_path);
    fs::rename(temp_path, final_path);
}

void write_gsadf_summary_csv_atomic(
    const string& output_dir,
    int lag,
    const vector<GsadfSummaryRow>& summary_rows
) {
    string filename = "GSADF_summary_lag" + to_string(lag) + ".csv";
    fs::path final_path = fs::path(output_dir) / filename;
    fs::path temp_path  = fs::path(output_dir) / (filename + ".tmp");

    ofstream fout(temp_path);
    if (!fout.is_open()) {
        throw runtime_error("Could not write temp GSADF summary output: " + temp_path.string());
    }

    fout << "market_name,market_type,input_file,lag,observations,min_window,first_date,last_date,gsadf,gsadf_critical_value,gsadf_reject\n";
    fout << setprecision(15);

    for (const auto& r : summary_rows) {
        fout << r.market_name << ","
             << r.market_type << ","
             << r.input_file << ","
             << r.lag << ","
             << r.observations << ","
             << r.min_window << ","
             << r.first_date << ","
             << r.last_date << ",";

        if (isfinite(r.gsadf)) fout << r.gsadf;
        fout << ",";

        if (isfinite(r.gsadf_critical_value)) fout << r.gsadf_critical_value;
        fout << ",";

        fout << bool_to_text(r.gsadf_reject) << "\n";
    }

    fout.close();
    if (fs::exists(final_path)) fs::remove(final_path);
    fs::rename(temp_path, final_path);
}

// ============================================================
// MAIN
// ============================================================
int main() {
    try {
        cout << "============================================================\n";
        cout << "MULTI-MARKET BSADF + GSADF PIPELINE\n";
        cout << "============================================================\n";
        cout << "Dataset root          : " << DATASET_ROOT_DIR << "\n";
        cout << "Output parent         : " << OUTPUT_PARENT_DIR << "\n";
        cout << "Monte Carlo sims      : " << MONTE_CARLO_SIMS << "\n";
        cout << "Sig level             : " << SIG_LEVEL << "\n";
        cout << "Min bubble duration   : " << MIN_BUBBLE_DURATION << "\n";
        cout << "Sort by date          : " << (SORT_BY_DATE_LEXICOGRAPHIC ? "YES" : "NO") << "\n";
        cout << "ADF lags              : ";
        for (size_t i = 0; i < ADF_LAG_LIST.size(); ++i) {
            if (i) cout << ", ";
            cout << ADF_LAG_LIST[i];
        }
        cout << "\n";
        cout << "============================================================\n\n";

        if (!fs::exists(DATASET_ROOT_DIR)) {
            throw runtime_error("Dataset root does not exist: " + DATASET_ROOT_DIR);
        }

        vector<string> subfolders = {"ETFS", "ExchangeRate", "crypto"};
        vector<MarketData> all_markets;

        for (const string& sub : subfolders) {
            fs::path folder = fs::path(DATASET_ROOT_DIR) / sub;
            vector<fs::path> files = list_csv_files(folder);

            cout << "Scanning folder: " << folder.string() << "\n";
            cout << "Found " << files.size() << " CSV file(s).\n";

            string market_type = folder_to_market_type(sub);
            for (const auto& f : files) {
                try {
                    all_markets.push_back(read_market_csv_first_two_columns(f, market_type));
                } catch (const exception& e) {
                    cerr << "Skipping file due to read error: " << f.string() << "\n";
                    cerr << "Reason: " << e.what() << "\n";
                }
            }
        }

        if (all_markets.empty()) {
            throw runtime_error("No valid market CSV files found in the dataset tree.");
        }

        string run_folder_name = "run_" + now_timestamp_for_folder();
        fs::path output_dir = fs::path(OUTPUT_PARENT_DIR) / run_folder_name;
        fs::create_directories(output_dir);

        cout << "\nTotal valid markets loaded: " << all_markets.size() << "\n";
        cout << "Run output folder: " << output_dir.string() << "\n\n";

        map<MCKey, vector<double>> bsadf_cv_cache;
        map<MCKey, double> gsadf_cv_cache;

        auto global_start = chrono::steady_clock::now();

        for (int lag : ADF_LAG_LIST) {
            cout << "\n============================================================\n";
            cout << "STARTING LAG = " << lag << "\n";
            cout << "============================================================\n";

            vector<GsadfSummaryRow> gsadf_summary_rows;

            for (size_t idx = 0; idx < all_markets.size(); ++idx) {
                auto market_start = chrono::steady_clock::now();

                MarketData md = all_markets[idx];

                cout << "------------------------------------------------------------\n";
                cout << "Market " << (idx + 1) << "/" << all_markets.size() << "\n";
                cout << "Name        : " << md.market_name << "\n";
                cout << "Type        : " << md.market_type << "\n";
                cout << "Input       : " << md.input_path << "\n";
                cout << "Lag         : " << lag << "\n";
                cout << "------------------------------------------------------------\n";

                int T = static_cast<int>(md.rows.size());
                int min_window = get_min_window(T);
                if (min_window < lag + 5) min_window = lag + 5;

                cout << "Observations: " << T << "\n";
                cout << "Min window  : " << min_window << "\n";

                vector<double> log_series;
                log_series.reserve(md.rows.size());
                for (const auto& r : md.rows) log_series.push_back(r.log_price);

                MCKey key{T, min_window, lag, MONTE_CARLO_SIMS, SIG_LEVEL};

                // ----------------------------
                // BSADF critical values
                // ----------------------------
                vector<double> bsadf_cv;
                auto it_b = bsadf_cv_cache.find(key);
                if (it_b != bsadf_cv_cache.end()) {
                    cout << "Reusing cached BSADF Monte Carlo critical values.\n";
                    bsadf_cv = it_b->second;
                } else {
                    cout << "Computing BSADF Monte Carlo critical values...\n";
                    bsadf_cv = compute_monte_carlo_bsadf_critical_values(
                        T,
                        min_window,
                        lag,
                        MONTE_CARLO_SIMS,
                        SIG_LEVEL,
                        "MC_BSADF_" + md.market_name + "_L" + to_string(lag)
                    );
                    bsadf_cv_cache[key] = bsadf_cv;
                }

                // ----------------------------
                // GSADF critical value
                // ----------------------------
                double gsadf_cv;
                auto it_g = gsadf_cv_cache.find(key);
                if (it_g != gsadf_cv_cache.end()) {
                    cout << "Reusing cached GSADF Monte Carlo critical value.\n";
                    gsadf_cv = it_g->second;
                } else {
                    cout << "Computing GSADF Monte Carlo critical value...\n";
                    gsadf_cv = compute_monte_carlo_gsadf_critical_value(
                        T,
                        min_window,
                        lag,
                        MONTE_CARLO_SIMS,
                        SIG_LEVEL,
                        "MC_GSADF_" + md.market_name + "_L" + to_string(lag)
                    );
                    gsadf_cv_cache[key] = gsadf_cv;
                }

                // ----------------------------
                // BSADF path
                // ----------------------------
                cout << "Computing BSADF path...\n";
                vector<double> bsadf = compute_bsadf_general(
                    log_series,
                    min_window,
                    lag,
                    "BSADF_" + md.market_name + "_L" + to_string(lag)
                );

                for (size_t i = 0; i < md.rows.size(); ++i) {
                    md.rows[i].bsadf = bsadf[i];
                    md.rows[i].critical_value = bsadf_cv[i];
                }

                write_bsadf_result_csv_atomic(
                    output_dir.string(),
                    md,
                    md.rows,
                    lag,
                    MIN_BUBBLE_DURATION
                );
                cout << "Wrote BSADF output.\n";

                // ----------------------------
                // GSADF summary row
                // ----------------------------
                cout << "Computing GSADF statistic...\n";
                double gsadf = compute_gsadf_general(
                    log_series,
                    min_window,
                    lag,
                    "GSADF_" + md.market_name + "_L" + to_string(lag)
                );

                GsadfSummaryRow row;
                row.market_name = md.market_name;
                row.market_type = md.market_type;
                row.input_file = fs::path(md.input_path).filename().string();
                row.lag = lag;
                row.observations = T;
                row.min_window = min_window;
                row.first_date = md.rows.front().date;
                row.last_date = md.rows.back().date;
                row.gsadf = gsadf;
                row.gsadf_critical_value = gsadf_cv;
                row.gsadf_reject = isfinite(gsadf) && isfinite(gsadf_cv) && (gsadf > gsadf_cv);

                gsadf_summary_rows.push_back(row);

                auto market_end = chrono::steady_clock::now();
                double elapsed = chrono::duration<double>(market_end - market_start).count();
                cout << "Finished market in " << format_seconds(elapsed) << ".\n\n";
            }

            write_gsadf_summary_csv_atomic(output_dir.string(), lag, gsadf_summary_rows);
            cout << "Wrote GSADF summary for lag " << lag << ".\n";
        }

        auto global_end = chrono::steady_clock::now();
        double total_elapsed = chrono::duration<double>(global_end - global_start).count();

        cout << "\n============================================================\n";
        cout << "ALL DONE\n";
        cout << "Output folder: " << output_dir.string() << "\n";
        cout << "Total elapsed: " << format_seconds(total_elapsed) << "\n";
        cout << "============================================================\n";

    } catch (const exception& e) {
        cerr << "\nFATAL ERROR:\n" << e.what() << "\n";
        return 1;
    }

    return 0;
}