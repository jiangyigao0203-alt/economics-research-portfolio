"""Download daily ETF price histories for the configured ticker universe.

Use yfinance over the fixed sample dates below, flatten returned column labels,
and write one CSV per ticker into dataset/ relative to the working directory.
Crypto and exchange-rate inputs require separate acquisition. Run from the
575 directory. Downloads start on import and require network access.
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

tickers = [
    "VOO","IVV","SPY","VTI","QQQ","VEA","VUG","IEFA","VTV","IEMG",
    "VXUS","SPYM","IWF","VWO","VGT","VIG","XLK","SCHD","ITOT","EFA",
    "ACWI","IDEV","IUSG","TQQQ","EEM","XLU","IUSV","XLC","FNDX","SOXX",
    "FNDF","MTUM","VOOG","XLY","RDVY","CGGR","EWJ","EMXC","DFIV","VYMI",
    "IYW","EWY","IWV","VONV","SPEM","VYM","IWD","QQQM","VT","IVW"
]

output_dir = Path("dataset")
output_dir.mkdir(exist_ok=True)

for t in tickers:
    print(f"Downloading {t}...")
    try:
        df = yf.download(t, start="1900-01-01", end="2026-04-11", interval="1d", auto_adjust=False)

        if df is None or df.empty:
            print(f"Empty data for {t}")
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Adj Close" in df.columns:
            price = df["Adj Close"]
        else:
            price = df["Close"]

        out = price.to_frame().reset_index()
        out.columns = ["date", "price"]

        file_path = output_dir / f"{t}.csv"
        out.to_csv(file_path, index=False)
        print(f"Saved {file_path}")

    except Exception as e:
        logging.getLogger(__name__).exception("Operation failed in %s", __file__)
        print(f"Error downloading {t}: {e}")

print("All done.")
