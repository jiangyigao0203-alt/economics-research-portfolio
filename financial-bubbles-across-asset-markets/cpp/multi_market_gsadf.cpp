/*
Whole-sample GSADF tests for a directory of asset-price series.

Each series is tested against a Gaussian random-walk null using 500 simulations
and the empirical 95th percentile. Standard C++ threads distribute regression
windows and simulations. Results are written to a timestamped market_summary.csv.
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
// Process one market-category directory of date/positive-price CSVs.
// The first two columns are date and price; later columns are ignored.
// Prices are logged internally. The regression and prefix sums are specialized
// to one lagged difference, so ADF_LAGS must remain 1 in this implementation.
// Monte Carlo draws calibrate the maximum statistic over all admissible windows.
// Critical values are reused for matching sample length and test configuration.

const string DATASET_DIR = "../data/markets/ETFS";
const string OUTPUT_PARENT_DIR = "../outputs/gsadf";

const int ADF_LAGS = 1;                   // fixed at 1
const int MONTE_CARLO_SIMS = 500;         // Null paths per calibration.
const double SIG_LEVEL = 0.95;            // 95% critical value
const bool SORT_BY_DATE_LEXICOGRAPHIC = false;
const int PROGRESS_REFRESH_MS = 1000;

// ============================================================
// DATA STRUCTURES
// ============================================================
struct MarketRow {
    string date;
    double raw_price = numeric_limits<double>::quiet_NaN();
    double log_price = numeric_limits<double>::quiet_NaN();
};

struct MarketData {
    string market_name;
    string input_path;
    vector<MarketRow> rows;
};

struct PrefixSumsLag1Full {
    // Regression rows r = 0..T-3 correspond to:
    // dep_r  = y[r+2] - y[r+1]
    // ylag_r = y[r+1]
    // dlag_r = y[r+1] - y[r]
    //
    // Regression:
    // dep = alpha + beta * ylag + phi * dlag + e

    vector<double> p1, py, pd, py2, pd2, pyd, pdep, pdep2, pydep, pddep;
};

struct ProgressState {
    atomic<long long> done{0};
    long long total = 1;
    atomic<bool> finished{false};
    string label;
};

struct MonteCarloCacheEntry {
    int T = 0;
    int min_window = 0;
    int lags = 0;
    int sims = 0;
    double sig_level = 0.0;
    double gsadf_critical_value = numeric_limits<double>::quiet_NaN();
};

struct MarketSummaryRow {
    string market_name;
    string input_file;
    int observations = 0;
    int min_window = 0;
    string first_date;
    string last_date;
    double gsadf = numeric_limits<double>::quiet_NaN();
    double gsadf_critical_value = numeric_limits<double>::quiet_NaN();
    bool gsadf_reject = false;
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

string bool_to_text(bool x) {
    return x ? "True" : "False";
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

int get_min_window(int T) {
    double r0 = 0.01 + 1.8 / sqrt(static_cast<double>(T));
    return max(10, static_cast<int>(floor(r0 * T)));
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

    cout << "\r" << left << setw(24) << label
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

// ============================================================
// MATRIX / OLS HELPERS FOR LAG-1 ADF
// ============================================================
// Regressor count k = 3: [1, ylag, dlag]
bool invert_3x3(const double A[3][3], double invA[3][3]) {
    double aug[3][6] = {
        {A[0][0], A[0][1], A[0][2], 1.0, 0.0, 0.0},
        {A[1][0], A[1][1], A[1][2], 0.0, 1.0, 0.0},
        {A[2][0], A[2][1], A[2][2], 0.0, 0.0, 1.0}
    };

    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int r = col + 1; r < 3; ++r) {
            if (fabs(aug[r][col]) > fabs(aug[pivot][col])) pivot = r;
        }
        if (fabs(aug[pivot][col]) < 1e-14) return false;

        if (pivot != col) {
            for (int j = 0; j < 6; ++j) swap(aug[col][j], aug[pivot][j]);
        }

        double div = aug[col][col];
        for (int j = 0; j < 6; ++j) aug[col][j] /= div;

        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            double factor = aug[r][col];
            for (int j = 0; j < 6; ++j) aug[r][j] -= factor * aug[col][j];
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) invA[i][j] = aug[i][j + 3];
    }
    return true;
}

PrefixSumsLag1Full build_prefix_sums_lag1_full(const vector<double>& y) {
    PrefixSumsLag1Full pref;
    int T = static_cast<int>(y.size());
    int m = max(0, T - 2);

    pref.p1.assign(m + 1, 0.0);
    pref.py.assign(m + 1, 0.0);
    pref.pd.assign(m + 1, 0.0);
    pref.py2.assign(m + 1, 0.0);
    pref.pd2.assign(m + 1, 0.0);
    pref.pyd.assign(m + 1, 0.0);
    pref.pdep.assign(m + 1, 0.0);
    pref.pdep2.assign(m + 1, 0.0);
    pref.pydep.assign(m + 1, 0.0);
    pref.pddep.assign(m + 1, 0.0);

    for (int r = 0; r < m; ++r) {
        double ylag = y[r + 1];
        double dlag = y[r + 1] - y[r];
        double dep  = y[r + 2] - y[r + 1];

        pref.p1[r + 1]     = pref.p1[r]     + 1.0;
        pref.py[r + 1]     = pref.py[r]     + ylag;
        pref.pd[r + 1]     = pref.pd[r]     + dlag;
        pref.py2[r + 1]    = pref.py2[r]    + ylag * ylag;
        pref.pd2[r + 1]    = pref.pd2[r]    + dlag * dlag;
        pref.pyd[r + 1]    = pref.pyd[r]    + ylag * dlag;
        pref.pdep[r + 1]   = pref.pdep[r]   + dep;
        pref.pdep2[r + 1]  = pref.pdep2[r]  + dep * dep;
        pref.pydep[r + 1]  = pref.pydep[r]  + ylag * dep;
        pref.pddep[r + 1]  = pref.pddep[r]  + dlag * dep;
    }

    return pref;
}

bool adf_tstat_lag1_window(const PrefixSumsLag1Full& pref, int L, int R, double& tstat_out) {
    double n = pref.p1[R] - pref.p1[L];
    if (n <= 3.0) return false;

    double sum_y    = pref.py[R]    - pref.py[L];
    double sum_d    = pref.pd[R]    - pref.pd[L];
    double sum_y2   = pref.py2[R]   - pref.py2[L];
    double sum_d2   = pref.pd2[R]   - pref.pd2[L];
    double sum_yd   = pref.pyd[R]   - pref.pyd[L];
    double sum_dep  = pref.pdep[R]  - pref.pdep[L];
    double sum_dep2 = pref.pdep2[R] - pref.pdep2[L];
    double sum_ydep = pref.pydep[R] - pref.pydep[L];
    double sum_ddep = pref.pddep[R] - pref.pddep[L];

    double XtX[3][3] = {
        {n,      sum_y,   sum_d},
        {sum_y,  sum_y2,  sum_yd},
        {sum_d,  sum_yd,  sum_d2}
    };
    double XtY[3] = {sum_dep, sum_ydep, sum_ddep};

    double invXtX[3][3];
    if (!invert_3x3(XtX, invXtX)) return false;

    double beta[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            beta[i] += invXtX[i][j] * XtY[j];
        }
    }

    double beta_XtY = beta[0] * XtY[0] + beta[1] * XtY[1] + beta[2] * XtY[2];
    double sse = sum_dep2 - beta_XtY;

    double dof = n - 3.0;
    if (dof <= 0.0) return false;
    if (sse < 0.0 && sse > -1e-10) sse = 0.0;
    if (sse < 0.0) return false;

    double s2 = sse / dof;
    double var_beta_y = s2 * invXtX[1][1];
    if (!(var_beta_y > 0.0)) return false;

    tstat_out = beta[1] / sqrt(var_beta_y);
    return isfinite(tstat_out);
}

// ============================================================
// FILE IO
// ============================================================
vector<fs::path> list_csv_files(const string& folder) {
    vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        string ext = to_lower_copy(entry.path().extension().string());
        if (ext == ".csv") files.push_back(entry.path());
    }
    sort(files.begin(), files.end());
    return files;
}

MarketData read_market_csv_first_two_columns(const fs::path& path) {
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
// GSADF COMPUTATION
// ============================================================
double compute_gsadf_lag1_multicore(
    const vector<double>& y,
    int min_window,
    const string& progress_label
) {
    int T = static_cast<int>(y.size());
    if (T < 6) return numeric_limits<double>::quiet_NaN();
    if (min_window < 5) min_window = 5;

    PrefixSumsLag1Full pref = build_prefix_sums_lag1_full(y);

    vector<int> endpoints;
    for (int e = min_window - 1; e < T; ++e) {
        endpoints.push_back(e);
    }

    ProgressState ps;
    ps.total = static_cast<long long>(endpoints.size());
    ps.label = progress_label;

    thread pth(progress_thread_runner, &ps);

    unsigned int n_threads = thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;

    atomic<size_t> next_idx(0);
    vector<thread> workers;

    double global_best = -numeric_limits<double>::infinity();
    mutex global_best_mutex;

    auto worker = [&]() {
        double local_best = -numeric_limits<double>::infinity();

        while (true) {
            size_t idx = next_idx.fetch_add(1);
            if (idx >= endpoints.size()) break;

            int e = endpoints[idx];
            int start_max = e - min_window + 1;

            for (int start = start_max; start >= 0; --start) {
                int L = start;
                int R = e - 1;   // exclusive
                if (R <= L) continue;

                double tstat = numeric_limits<double>::quiet_NaN();
                if (adf_tstat_lag1_window(pref, L, R, tstat)) {
                    if (tstat > local_best) local_best = tstat;
                }
            }

            ps.done.fetch_add(1);
        }

        lock_guard<mutex> lock(global_best_mutex);
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

string make_cache_key(int T, int min_window, int lags, int sims, double sig_level) {
    ostringstream oss;
    oss << "T=" << T
        << "|W=" << min_window
        << "|L=" << lags
        << "|S=" << sims
        << "|Q=" << fixed << setprecision(6) << sig_level;
    return oss.str();
}

double compute_monte_carlo_gsadf_critical_value(
    int T,
    int min_window,
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

            uint64_t seed = 123456789ULL
                          + static_cast<uint64_t>(sim) * 7919ULL
                          + static_cast<uint64_t>(T) * 104729ULL;

            vector<double> y = simulate_random_walk(T, seed);
            PrefixSumsLag1Full pref = build_prefix_sums_lag1_full(y);

            double gsadf_best = -numeric_limits<double>::infinity();

            for (int e = min_window - 1; e < T; ++e) {
                int start_max = e - min_window + 1;

                for (int start = start_max; start >= 0; --start) {
                    int L = start;
                    int R = e - 1;   // exclusive
                    if (R <= L) continue;

                    double tstat = numeric_limits<double>::quiet_NaN();
                    if (adf_tstat_lag1_window(pref, L, R, tstat)) {
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
// OUTPUT
// ============================================================
void write_summary_csv_atomic(
    const string& output_dir,
    const vector<MarketSummaryRow>& summary_rows
) {
    fs::path final_path = fs::path(output_dir) / "market_summary.csv";
    fs::path temp_path  = fs::path(output_dir) / "market_summary.tmp.csv";

    ofstream fout(temp_path);
    if (!fout.is_open()) {
        throw runtime_error("Could not write temp summary output: " + temp_path.string());
    }

    fout << "market_name,input_file,observations,min_window,first_date,last_date,gsadf,gsadf_critical_value,gsadf_reject\n";
    fout << setprecision(15);

    for (const auto& r : summary_rows) {
        fout << r.market_name << ","
             << r.input_file << ","
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
        cout << "MULTI-MARKET GSADF PIPELINE (C++, ADF LAG = 1, GSADF ONLY)\n";
        cout << "============================================================\n";
        cout << "Dataset folder        : " << DATASET_DIR << "\n";
        cout << "Output parent folder  : " << OUTPUT_PARENT_DIR << "\n";
        cout << "ADF lags              : " << ADF_LAGS << "\n";
        cout << "Monte Carlo sims      : " << MONTE_CARLO_SIMS << "\n";
        cout << "Sig level             : " << SIG_LEVEL << "\n";
        cout << "Sort lexicographically: " << (SORT_BY_DATE_LEXICOGRAPHIC ? "YES" : "NO") << "\n";
        cout << "============================================================\n\n";

        if (!fs::exists(DATASET_DIR)) {
            throw runtime_error("Dataset folder does not exist: " + DATASET_DIR);
        }

        vector<fs::path> files = list_csv_files(DATASET_DIR);
        if (files.empty()) {
            throw runtime_error("No CSV files found in dataset folder.");
        }

        string run_folder_name = "run_" + now_timestamp_for_folder();
        fs::path output_dir = fs::path(OUTPUT_PARENT_DIR) / run_folder_name;
        fs::create_directories(output_dir);

        cout << "Found " << files.size() << " market file(s).\n";
        cout << "Run output folder: " << output_dir.string() << "\n\n";

        vector<pair<string, MonteCarloCacheEntry>> mc_cache;
        vector<MarketSummaryRow> all_summary_rows;

        auto global_start = chrono::steady_clock::now();

        for (size_t file_idx = 0; file_idx < files.size(); ++file_idx) {
            auto market_start = chrono::steady_clock::now();

            cout << "------------------------------------------------------------\n";
            cout << "MARKET " << (file_idx + 1) << "/" << files.size() << "\n";
            cout << "File   : " << files[file_idx].filename().string() << "\n";
            cout << "------------------------------------------------------------\n";

            MarketData md = read_market_csv_first_two_columns(files[file_idx]);
            int T = static_cast<int>(md.rows.size());
            int min_window = get_min_window(T);

            cout << "Market name           : " << md.market_name << "\n";
            cout << "Usable observations   : " << T << "\n";
            cout << "Minimum window        : " << min_window << "\n";

            vector<double> log_series;
            log_series.reserve(md.rows.size());
            for (const auto& r : md.rows) log_series.push_back(r.log_price);

            string key = make_cache_key(T, min_window, ADF_LAGS, MONTE_CARLO_SIMS, SIG_LEVEL);

            const MonteCarloCacheEntry* mc_ptr = nullptr;
            bool cache_hit = false;
            for (auto& kv : mc_cache) {
                if (kv.first == key) {
                    mc_ptr = &kv.second;
                    cache_hit = true;
                    break;
                }
            }

            if (cache_hit) {
                cout << "Monte Carlo cache     : HIT\n";
            } else {
                cout << "Monte Carlo cache     : MISS\n";

                double cv = compute_monte_carlo_gsadf_critical_value(
                    T, min_window, MONTE_CARLO_SIMS, SIG_LEVEL,
                    "Monte Carlo: " + md.market_name
                );

                MonteCarloCacheEntry entry;
                entry.T = T;
                entry.min_window = min_window;
                entry.lags = ADF_LAGS;
                entry.sims = MONTE_CARLO_SIMS;
                entry.sig_level = SIG_LEVEL;
                entry.gsadf_critical_value = cv;

                mc_cache.push_back({key, std::move(entry)});
                mc_ptr = &mc_cache.back().second;
            }

            double gsadf = compute_gsadf_lag1_multicore(
                log_series, min_window, "GSADF: " + md.market_name
            );

            MarketSummaryRow sr;
            sr.market_name = md.market_name;
            sr.input_file = files[file_idx].filename().string();
            sr.observations = T;
            sr.min_window = min_window;
            sr.first_date = md.rows.front().date;
            sr.last_date = md.rows.back().date;
            sr.gsadf = gsadf;
            sr.gsadf_critical_value = mc_ptr->gsadf_critical_value;
            sr.gsadf_reject = isfinite(gsadf)
                           && isfinite(mc_ptr->gsadf_critical_value)
                           && (gsadf > mc_ptr->gsadf_critical_value);

            all_summary_rows.push_back(sr);

            auto market_end = chrono::steady_clock::now();
            double market_elapsed = chrono::duration<double>(market_end - market_start).count();

            auto global_now = chrono::steady_clock::now();
            double global_elapsed = chrono::duration<double>(global_now - global_start).count();
            double avg_per_market = global_elapsed / static_cast<double>(file_idx + 1);
            double eta = avg_per_market * static_cast<double>(files.size() - file_idx - 1);

            cout << "GSADF statistic       : ";
            if (isfinite(gsadf)) cout << gsadf << "\n";
            else cout << "NaN\n";

            cout << "GSADF critical value  : ";
            if (isfinite(mc_ptr->gsadf_critical_value)) cout << mc_ptr->gsadf_critical_value << "\n";
            else cout << "NaN\n";

            cout << "GSADF reject          : " << (sr.gsadf_reject ? "YES" : "NO") << "\n";
            cout << "Market time           : " << format_seconds(market_elapsed) << "\n";
            cout << "Overall progress      : " << (file_idx + 1) << "/" << files.size()
                 << " | ETA " << format_seconds(eta) << "\n\n";
        }

        write_summary_csv_atomic(output_dir.string(), all_summary_rows);

        auto global_end = chrono::steady_clock::now();
        double total_elapsed = chrono::duration<double>(global_end - global_start).count();

        cout << "============================================================\n";
        cout << "ALL MARKETS COMPLETE\n";
        cout << "============================================================\n";
        cout << "Output folder         : " << output_dir.string() << "\n";
        cout << "Summary file          : " << (output_dir / "market_summary.csv").string() << "\n";
        cout << "Total elapsed         : " << format_seconds(total_elapsed) << "\n";

    } catch (const exception& e) {
        cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}