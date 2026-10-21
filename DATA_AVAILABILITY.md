# Data availability

This repository shares source code and selected aggregate statistical outputs. It does not distribute raw or observation-level datasets.

| Project | Required external inputs | Access / preparation |
| --- | --- | --- |
| 525 | Compustat fundamentals and security data; CRSP prices and market/risk-free returns | Use your own authorized WRDS access and review the download scripts |
| 525 | Monetary-policy surprises / shock series; optional NYSE market-equity breakpoints | Obtain the original sources, establish citations and permissions, and match the required schemas |
| 575 | Cryptocurrency, ETF, exchange-rate, and valuation series; derived BSADF files | Review acquisition scripts and source terms; some source files and derived folders are absent |

Do not add WRDS credentials, `.pgpass`, `.env`, passwords, or access tokens. Raw inputs and generated panels remain local. The `.gitignore` permits only the explicitly selected result CSV files; adding a future dataset requires a deliberate change.

Data-source URLs, scholarly references, and the exact configurations underlying historical outputs still need to be completed by the author. No license is assigned to third-party datasets by publishing this code.
