"""Download CRSP daily prices matched to the Compustat firm sample.

Read the monthly Compustat panel, obtain CCM links through WRDS, and download
prices in batches over the configured dates. The output is a firm-date price
panel. Authorized WRDS/CRSP access is required; data are not bundled here.
Run from the project directory. Importing this script starts the download workflow.
"""

import math
import time

import pandas as pd
import wrds

# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = "compustat_monthly_panel_from_quarterly_2000_2024.csv"
OUTPUT_FILE = "gvkey_crsp_daily_price_matched.csv"

START_DATE = "1999-11-30"
END_DATE   = "2025-12-31"

GVKEY_COL = "gvkey"

# Preserve company names when the input panel contains them.
NAME_COL_CANDIDATES = ["conm", "company_name", "name"]

# Batch size for CRSP download
PERMNO_BATCH_SIZE = 200


# ============================================================
# HELPERS
# ============================================================
def log(msg: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def batched_list(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


# ============================================================
# STEP 0: LOAD INPUT FILE
# ============================================================
t0 = time.time()
log("Loading input file...")

df = pd.read_csv(INPUT_FILE)

if GVKEY_COL not in df.columns:
    raise ValueError(f"Column '{GVKEY_COL}' not found in input file.")

df[GVKEY_COL] = df[GVKEY_COL].astype(str).str.zfill(6)

# Keep a gvkey -> company name map if available
name_col = None
for c in NAME_COL_CANDIDATES:
    if c in df.columns:
        name_col = c
        break

if name_col is not None:
    gvkey_name_map = (
        df[[GVKEY_COL, name_col]]
        .dropna(subset=[GVKEY_COL])
        .drop_duplicates(subset=[GVKEY_COL])
        .copy()
    )
    gvkey_name_map = gvkey_name_map.rename(columns={name_col: "company_name"})
else:
    gvkey_name_map = pd.DataFrame({GVKEY_COL: sorted(df[GVKEY_COL].dropna().unique())})
    gvkey_name_map["company_name"] = pd.NA

gvkeys = sorted(df[GVKEY_COL].dropna().unique().tolist())

if not gvkeys:
    raise ValueError("No gvkeys found in input file.")

gvkey_sql = ",".join([f"'{g}'" for g in gvkeys])

log("Loaded input file successfully.")
log(f"Rows in input file: {len(df):,}")
log(f"Unique gvkeys: {len(gvkeys):,}")
if name_col is not None:
    log(f"Using company name column: {name_col}")
else:
    log("No company name column found; output will have blank company_name.")
log(f"Step 0 completed in {time.time() - t0:.2f} seconds.")


# ============================================================
# STEP 1: CONNECT TO WRDS
# ============================================================
t1 = time.time()
log("Connecting to WRDS...")
conn = wrds.Connection()
log("Connected to WRDS.")
log(f"Step 1 completed in {time.time() - t1:.2f} seconds.")


# ============================================================
# STEP 2: QUERY CCM LINK TABLE
# ============================================================
t2 = time.time()
log("Querying CCM link table...")

ccm = conn.raw_sql(f"""
    select
        gvkey,
        lpermno as permno,
        linktype,
        linkprim,
        linkdt,
        linkenddt
    from crsp.ccmxpf_linktable
    where gvkey in ({gvkey_sql})
      and lpermno is not null
      and linktype in ('LU','LC')
      and linkprim in ('P','C')
""", date_cols=["linkdt", "linkenddt"])

if ccm.empty:
    raise ValueError("No CCM links found. Verify WRDS subscriptions and firm identifiers.")

ccm["gvkey"] = ccm["gvkey"].astype(str).str.zfill(6)
ccm["permno"] = ccm["permno"].astype("Int64")

ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])
ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

# Restrict links to the configured sample window.
ccm = ccm[
    (ccm["linkenddt"] >= pd.Timestamp(START_DATE)) &
    (ccm["linkdt"] <= pd.Timestamp(END_DATE))
].copy()

if ccm.empty:
    raise ValueError("CCM links exist, but none overlap the configured date range.")

log(f"CCM rows after filtering: {len(ccm):,}")
log(f"Unique gvkeys matched in CCM: {ccm['gvkey'].nunique():,}")
log(f"Unique permnos from CCM: {ccm['permno'].nunique():,}")
log(f"Step 2 completed in {time.time() - t2:.2f} seconds.")


# ============================================================
# STEP 3: PREPARE PERMNO BATCHES
# ============================================================
t3 = time.time()
permnos = sorted(ccm["permno"].dropna().astype(int).unique().tolist())

if not permnos:
    raise ValueError("No valid permnos found after CCM filtering.")

num_batches = math.ceil(len(permnos) / PERMNO_BATCH_SIZE)

log("Preparing CRSP daily price download batches...")
log(f"Total permnos: {len(permnos):,}")
log(f"Batch size: {PERMNO_BATCH_SIZE:,}")
log(f"Number of batches: {num_batches:,}")
log(f"Step 3 completed in {time.time() - t3:.2f} seconds.")


# ============================================================
# STEP 4: DOWNLOAD CRSP DAILY PRICE IN BATCHES
# ============================================================
t4 = time.time()
log("Starting batched CRSP daily price download...")

daily_parts = []
total_rows_downloaded = 0

for batch_idx, batch_permnos in enumerate(batched_list(permnos, PERMNO_BATCH_SIZE), start=1):
    batch_sql = ",".join(map(str, batch_permnos))
    batch_start = time.time()

    log(f"Starting batch {batch_idx}/{num_batches} with {len(batch_permnos):,} permnos...")

    tmp = conn.raw_sql(f"""
        select
            permno,
            date,
            prc
        from crsp.dsf
        where permno in ({batch_sql})
          and date between '{START_DATE}' and '{END_DATE}'
    """, date_cols=["date"])

    batch_rows = len(tmp)
    total_rows_downloaded += batch_rows

    log(
        f"Finished batch {batch_idx}/{num_batches}: "
        f"{batch_rows:,} rows downloaded "
        f"in {time.time() - batch_start:.2f} seconds."
    )

    if batch_rows > 0:
        daily_parts.append(tmp)

if not daily_parts:
    raise ValueError("No daily CRSP price data returned across all batches.")

daily = pd.concat(daily_parts, ignore_index=True)

log("Finished all CRSP batches.")
log(f"Total daily rows downloaded: {total_rows_downloaded:,}")
log(f"Combined daily DataFrame rows: {len(daily):,}")
log(f"Step 4 completed in {time.time() - t4:.2f} seconds.")


# ============================================================
# STEP 5: CLEAN DAILY PRICE
# ============================================================
t5 = time.time()
log("Cleaning daily price data...")

daily["prc"] = pd.to_numeric(daily["prc"], errors="coerce")
daily["price"] = daily["prc"].abs()

daily = daily.dropna(subset=["price", "date", "permno"]).copy()

log(f"Daily rows after cleaning: {len(daily):,}")
log(f"Step 5 completed in {time.time() - t5:.2f} seconds.")


# ============================================================
# STEP 6: MERGE DAILY PRICE BACK TO GVKEY USING CCM LINK DATES
# ============================================================
t6 = time.time()
log("Merging daily price data with CCM links...")

matched = daily.merge(ccm, on="permno", how="inner")
log(f"Rows after merge on permno: {len(matched):,}")

matched = matched[
    (matched["date"] >= matched["linkdt"]) &
    (matched["date"] <= matched["linkenddt"])
].copy()

log(f"Rows after enforcing CCM link date validity: {len(matched):,}")

if matched.empty:
    raise ValueError("Matched dataset is empty after applying link-date filters.")

matched = matched[
    ["gvkey", "permno", "date", "price", "linktype", "linkprim", "linkdt", "linkenddt"]
].sort_values(["gvkey", "date", "permno"]).reset_index(drop=True)

log(f"Step 6 completed in {time.time() - t6:.2f} seconds.")


# ============================================================
# STEP 7: HANDLE MULTIPLE PERMNOs FOR SAME GVKEY-DATE
# ============================================================
t7 = time.time()
log("Resolving duplicate gvkey-date matches...")

dup_before = matched.duplicated(subset=["gvkey", "date"], keep=False).sum()
log(f"Duplicate gvkey-date rows before tie-break: {dup_before:,}")

matched = matched.sort_values(["gvkey", "date", "permno"]).drop_duplicates(
    subset=["gvkey", "date"], keep="first"
)

dup_after = matched.duplicated(subset=["gvkey", "date"], keep=False).sum()
log(f"Duplicate gvkey-date rows after tie-break: {dup_after:,}")

log(f"Final matched rows: {len(matched):,}")
log(f"Unique gvkeys in final matched data: {matched['gvkey'].nunique():,}")
log(f"Step 7 completed in {time.time() - t7:.2f} seconds.")


# ============================================================
# STEP 8: ADD COMPANY NAME FROM INPUT FILE
# ============================================================
t8 = time.time()
log("Adding company names from input file...")

matched = matched.merge(gvkey_name_map, on="gvkey", how="left")

matched = matched[
    [
        "gvkey",
        "company_name",
        "permno",
        "date",
        "price",
        "linktype",
        "linkprim",
        "linkdt",
        "linkenddt",
    ]
].sort_values(["gvkey", "date"]).reset_index(drop=True)

log(f"Step 8 completed in {time.time() - t8:.2f} seconds.")


# ============================================================
# STEP 9: SAVE OUTPUT
# ============================================================
t9 = time.time()
log("Saving output file...")

matched.to_csv(OUTPUT_FILE, index=False)

log(f"Saved output to: {OUTPUT_FILE}")
log(f"Step 9 completed in {time.time() - t9:.2f} seconds.")


# ============================================================
# FINAL SUMMARY
# ============================================================
log("All steps completed successfully.")
log(f"Total runtime: {time.time() - t0:.2f} seconds.")
print(matched.head())