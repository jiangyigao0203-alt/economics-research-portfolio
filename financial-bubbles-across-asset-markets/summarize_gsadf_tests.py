"""Aggregate per-series GSADF tests into market-category rejection statistics."""

from pathlib import Path

import pandas as pd

# =========================
# SETTINGS
# =========================
INPUT_FOLDER = Path(__file__).resolve().parent / "results"

FILES = {
    "Crypto": "gsadf_crypto.csv",
    "ETFS": "gsadf_etfs.csv",
    "ExchangeRate": "gsadf_exchange_rates.csv",
}

# =========================
# MAIN LOGIC
# =========================
def summarize_file(file_path, market_type):
    df = pd.read_csv(file_path)

    # ensure boolean
    df["gsadf_reject"] = df["gsadf_reject"].astype(bool)

    total = len(df)
    reject_count = df["gsadf_reject"].sum()
    avg_gsadf = df["gsadf"].mean()

    return {
        "market_type": market_type,
        "n_series": total,
        "reject_count": int(reject_count),
        "reject_rate": reject_count / total if total > 0 else None,
        "avg_gsadf": avg_gsadf,
    }


def main():
    results = []

    for market, filename in FILES.items():
        file_path = INPUT_FOLDER / filename

        if not file_path.exists():
            print(f"⚠️ Missing file: {file_path}")
            continue

        summary = summarize_file(file_path, market)
        results.append(summary)

    if not results:
        print("No valid files found.")
        return

    summary_df = pd.DataFrame(results)

    # =========================
    # PRINT RESULTS
    # =========================
    print("\n" + "=" * 60)
    print("GSADF SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # =========================
    # OPTIONAL: SAVE OUTPUT
    # =========================
    output_file = Path(__file__).resolve().parent / "outputs" / "gsadf_market_summary.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary to: {output_file}")


if __name__ == "__main__":
    main()