# Running the S&P 500 earnings-call pipeline on your own computer

`sp500_pipeline.py` runs on Windows, Mac, Linux, or a cloud virtual machine (GCP). It downloads the data from WRDS, builds one row per earnings call, and saves everything in a folder on your machine.

## What you need

| Item | Recommended |
|---|---|
| Python | 3.10 – 3.12 ([python.org](https://www.python.org/downloads/) or Anaconda) |
| Memory (RAM) | 16 GB for all S&P 500 (8 GB is fine for a few companies) |
| Free disk space | about 10 GB for all S&P 500 |
| Internet | stable connection; the first full download takes a few hours |
| WRDS | your username, password and Duo on your phone |

## 1. Put the files in one folder

Create a folder, for example `Documents/earnings`, and put these files in it:
- `sp500_pipeline.py`
- `requirements.txt`
- optional: the Loughran-McDonald Master Dictionary CSV (for tone and hedging features)

## 2. Open a terminal in that folder

- **Windows:** open the folder in File Explorer, click the address bar, type `cmd`, press Enter.
- **Mac:** open Terminal and type `cd ` (with a space), drag the folder onto the window, press Enter.

## 3. Install the packages (once)

```
python -m venv venv
```
Then activate it:
- Windows: `venv\Scripts\activate`
- Mac / Linux: `source venv/bin/activate`

```
pip install -r requirements.txt
```

## 4. Connect to WRDS once (saves your password)

```
python -c "import wrds; wrds.Connection(wrds_username='bhuvighosh3')"
```
Enter your password, approve the Duo push, and answer **y** when asked to create a `.pgpass` file. Later runs will not ask for the password again (Duo may still ask).

## 5. Run

**All S&P 500 companies:**
```
python sp500_pipeline.py
```

**Useful options**

| Command | What it does |
|---|---|
| `--folder D:/earnings_data` | Save the data somewhere else (default: your home folder) |
| `--stages panel targets text final` | Rebuild the tables from data already downloaded (no WRDS needed) |
| `--start-year 2012` | Start from a later year |
| `--help` | Show all options |

**If it stops** (lost internet, computer asleep, WRDS timeout): run the same command again. Everything already downloaded is skipped. Keep the computer from sleeping during the first full run.

## 6. Results

Saved in `earnings_sp500` in your home folder (or in `--folder`):

| File | Contents |
|---|---|
| `model_table.parquet` / `model_table.csv` | One row per call: IDs, sector, Day 0, targets, controls, language features, `sample` (train / holdout) |
| `model_table_with_text.parquet` | The same plus transcript text (`prepared_text`, `qa_text`) |
| `data_report.txt` | Summary: calls per sample, companies, missing values, checks |
| `raw/` | Downloaded WRDS data (reused on the next run) |

Open the results in Python or Jupyter:
```python
import pandas as pd
m = pd.read_parquet("earnings_sp500/model_table.parquet")
train = m[m["sample"] == "train"]       # 2010-2022
test  = m[m["sample"] == "holdout"]     # 2023-2024, only companies that are also in train
```

To add tone and hedging later: put the Loughran-McDonald CSV in the folder and run
`python sp500_pipeline.py --stages text final`.

## Running on Google Cloud (GCP) instead

Same steps on a Compute Engine virtual machine:

1. Create a VM: **e2-standard-4** (16 GB RAM) with a **50 GB** disk, Ubuntu.
2. Connect with the **SSH** button in the console.
3. Install Python tools: `sudo apt update && sudo apt install -y python3-venv`
4. Upload the files (SSH window → **Upload file**), then follow steps 3–5 above.
5. Run inside `tmux` so it keeps going if the browser window closes:
   ```
   tmux new -s pipeline
   python sp500_pipeline.py
   ```
   Detach with `Ctrl+B` then `D`; come back later with `tmux attach -t pipeline`.
6. Download the results with SSH window → **Download file**, then **stop the VM** so it doesn't keep costing money.

> **Data license:** WRDS data is licensed to you. Keep it on private machines and storage only. If your WRDS account is an academic license, check with your WRDS representative before storing licensed data on a cloud provider.
