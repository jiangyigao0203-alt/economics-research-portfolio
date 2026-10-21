# Economics research portfolio

Two Python research projects on empirical asset pricing and financial econometrics.

| Project | Research focus | Methods in the code |
| --- | --- | --- |
| [525: Equity duration and monetary-policy shocks](525/README.md) | How equity-return responses to monetary-policy and information shocks vary with implied equity duration | WRDS data preparation, cash-flow duration estimates, event returns, fixed-effects panel regressions, duration-bin interactions |
| [575: Financial bubbles and volatility](575/README.md) | Detecting explosive price episodes and studying their relationship with volatility | Recursive right-tailed ADF statistics, Monte Carlo critical values, GARCH estimation, episode regressions, logistic classification |

## Start here

Each project README maps the scripts to their purpose and explains data requirements and known reproduction gaps. The `results/` folders contain selected aggregate outputs from existing local runs, not newly replicated estimates. They illustrate the work; they are not evidence that the current code reproduces every saved output.

## Environment

Python 3.11 or later is required by scripts using `datetime.UTC`. From this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dependency list was inferred from imports. Versions are not pinned, and installation and full empirical execution have not been validated. Run each script from its project folder after providing its inputs and reviewing its SETTINGS or CONFIG block. Some scripts perform downloads, estimation, file creation, or plotting immediately when run or imported.

## Data and reproducibility

Raw data, firm-level panels, credentials, virtual environments, and large generated files are excluded. The collection scripts allow readers with the relevant data access to inspect the acquisition process. WRDS-based inputs require the reader's own authorized access. The origin and redistribution permissions of external inputs must be established before adding data.

See [file selection](FILE_SELECTION.md), [data availability](DATA_AVAILABILITY.md), and [verification notes](VERIFICATION.md). This is a documented research-code snapshot with known manual configuration steps, rather than a one-command replication package.
