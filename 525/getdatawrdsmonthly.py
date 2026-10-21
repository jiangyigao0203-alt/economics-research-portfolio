from pathlib import Path

import pandas as pd
import wrds

# ============================================================
# SETTINGS
# ============================================================
START_FYEARQ = 2000
END_FYEARQ = 2024

GVKEY_LIST = None
# Example:
# GVKEY_LIST = ["001004", "012141", "064768"]

OUTPUT_FILE = "compustat_monthly_panel_from_quarterly_2000_2024.csv"


def connect_wrds():
    return wrds.Connection()


def _build_gvkey_filter(alias, gvkey_list=None):
    if gvkey_list:
        quoted = ",".join([f"'{x}'" for x in gvkey_list])
        return f" and {alias}.gvkey in ({quoted}) "
    return ""


def fetch_compustat_quarterly_fundamentals(conn, start_fyearq, end_fyearq, gvkey_list=None):
    """
    Fetch one extra prior fiscal year so the first sample months can still
    be matched to a prior available quarter after we shift quarters forward.
    """
    gvkey_filter = _build_gvkey_filter("a", gvkey_list)
    fetch_start = int(start_fyearq) - 1

    sql = f"""
        select
            a.gvkey,
            b.conm,
            a.datadate,
            a.fyearq,
            a.fqtr,
            a.saleq,
            a.ibq,
            a.seqq,
            a.prccq,
            a.cshoq
        from comp.fundq as a
        left join comp.company as b
            on a.gvkey = b.gvkey
        where a.indfmt = 'INDL'
          and a.datafmt = 'STD'
          and a.popsrc = 'D'
          and a.consol = 'C'
          and a.fyearq between {fetch_start} and {int(end_fyearq)}
          {gvkey_filter}
        order by a.gvkey, a.datadate
    """

    return conn.raw_sql(sql, date_cols=["datadate"])


def fetch_compustat_monthly_security(conn, start_fyearq, end_fyearq, gvkey_list=None):
    gvkey_filter = _build_gvkey_filter("s", gvkey_list)

    start_date = f"{int(start_fyearq)}-01-01"
    end_date = f"{int(end_fyearq)}-12-31"

    sql = f"""
        select
            s.gvkey,
            s.datadate,
            s.prccm,
            s.cshom
        from comp.secm as s
        where s.datadate between '{start_date}' and '{end_date}'
          {gvkey_filter}
        order by s.gvkey, s.datadate
    """

    return conn.raw_sql(sql, date_cols=["datadate"])


def expand_quarterly_to_monthly(df_quarterly):
    """
    For each quarterly row, create 3 monthly rows:
    the quarter-end month and the NEXT two month-ends.

    Example:
        2000-08-31 -> 2000-08-31, 2000-09-30, 2000-10-31
        2000-11-30 -> 2000-11-30, 2000-12-31, 2001-01-31

    This is the key fix versus the old version.
    """
    rows = []

    for _, row in df_quarterly.iterrows():
        q_end = pd.Timestamp(row["datadate"]) + pd.offsets.MonthEnd(0)

        month_list = [
            q_end,
            q_end + pd.offsets.MonthEnd(1),
            q_end + pd.offsets.MonthEnd(2),
        ]

        for month_end in month_list:
            rows.append({
                "gvkey": row["gvkey"],
                "conm": row["conm"],
                "quarterly_datadate": row["datadate"],
                "fyearq": row["fyearq"],
                "fqtr": row["fqtr"],
                "month_date": month_end,
                "saleq": row["saleq"],
                "ibq": row["ibq"],
                "seqq": row["seqq"],
                "prccq_quarterly": row["prccq"],
                "cshoq_quarterly": row["cshoq"],
            })

    out = pd.DataFrame(rows)

    out["gvkey"] = pd.to_numeric(out["gvkey"], errors="coerce")
    out = out.dropna(subset=["gvkey", "month_date", "quarterly_datadate"]).copy()
    out["gvkey"] = out["gvkey"].astype("int64")

    out = out.sort_values(["gvkey", "month_date", "quarterly_datadate"]).reset_index(drop=True)

    # If duplicate gvkey-month rows exist, keep the latest available quarter
    out = out.drop_duplicates(subset=["gvkey", "month_date"], keep="last").reset_index(drop=True)

    return out


def merge_monthly_security(monthly_panel, secm):
    secm = secm.copy()

    secm["gvkey"] = pd.to_numeric(secm["gvkey"], errors="coerce")
    secm = secm.dropna(subset=["gvkey", "datadate"]).copy()
    secm["gvkey"] = secm["gvkey"].astype("int64")

    secm["month_date"] = pd.to_datetime(secm["datadate"]) + pd.offsets.MonthEnd(0)

    secm = secm.rename(
        columns={
            "datadate": "security_datadate",
            "prccm": "prcc_m",
            "cshom": "csho_m_raw",
        }
    )

    # csho_m from secm appears to be raw shares here; convert to millions
    secm["csho_m"] = secm["csho_m_raw"] / 1_000_000.0

    out = monthly_panel.merge(
        secm[["gvkey", "month_date", "security_datadate", "prcc_m", "csho_m_raw", "csho_m"]],
        on=["gvkey", "month_date"],
        how="left",
    )

    # Use true monthly values when available, otherwise quarterly fallback
    out["prcc_used"] = out["prcc_m"].combine_first(out["prccq_quarterly"])
    out["csho_used"] = out["csho_m"].combine_first(out["cshoq_quarterly"])

    # Market cap in millions of dollars
    out["market_cap"] = out["prcc_used"].abs() * out["csho_used"]

    return out


def main():
    print("Connecting to WRDS...")
    conn = connect_wrds()

    try:
        print("Downloading Compustat quarterly fundamentals...")
        quarterly = fetch_compustat_quarterly_fundamentals(
            conn=conn,
            start_fyearq=START_FYEARQ,
            end_fyearq=END_FYEARQ,
            gvkey_list=GVKEY_LIST,
        )
        print(f"Quarterly rows downloaded: {len(quarterly):,}")

        print("Downloading Compustat monthly security data...")
        secm = fetch_compustat_monthly_security(
            conn=conn,
            start_fyearq=START_FYEARQ,
            end_fyearq=END_FYEARQ,
            gvkey_list=GVKEY_LIST,
        )
        print(f"Monthly security rows downloaded: {len(secm):,}")

        print("Expanding quarterly data to monthly rows...")
        monthly_panel = expand_quarterly_to_monthly(quarterly)
        print(f"Expanded monthly rows: {len(monthly_panel):,}")

        print("Merging monthly security data...")
        final = merge_monthly_security(monthly_panel, secm)

        # Keep only desired sample months
        final = final[
            (final["month_date"] >= pd.Timestamp(f"{START_FYEARQ}-01-01")) &
            (final["month_date"] <= pd.Timestamp(f"{END_FYEARQ}-12-31"))
        ].copy()

        final = final.sort_values(["gvkey", "month_date"]).reset_index(drop=True)

        final.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved monthly panel to: {Path(OUTPUT_FILE).resolve()}")

        monthly_price_coverage = final["prcc_m"].notna().mean()
        monthly_shares_coverage = final["csho_m"].notna().mean()

        print(f"Rows saved: {len(final):,}")
        print(f"Monthly price coverage: {monthly_price_coverage:.2%}")
        print(f"Monthly shares coverage: {monthly_shares_coverage:.2%}")

        cols_to_show = [
            "gvkey",
            "conm",
            "fyearq",
            "fqtr",
            "month_date",
            "quarterly_datadate",
            "saleq",
            "ibq",
            "seqq",
            "prcc_m",
            "csho_m_raw",
            "csho_m",
            "prcc_used",
            "csho_used",
            "market_cap",
        ]
        print(final[cols_to_show].head(15))

    finally:
        conn.close()


if __name__ == "__main__":
    main()