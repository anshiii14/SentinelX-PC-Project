# SentinelX — Parallel Hybrid Network Intrusion Detection System

A reproducible academic prototype based on the project report. It uses real CIC-IDS2017 labelled flow records, Random Forest for known attack classification, Isolation Forest for novelty/anomaly detection, and CPU multiprocessing for parallel model experiments and traffic-batch scoring.

## Important experimental design

- `PortScan` is held out from supervised Random Forest training.
- Random Forest learns the remaining available labelled classes.
- Isolation Forest is trained only on BENIGN training flows.
- PortScan is evaluated separately as a previously unseen category.
- `Unknown Anomaly` means suspicious/unusual, **not a confirmed PortScan identity**.

## Recommended environment

Use VS Code on Windows, macOS, or Linux. A multicore CPU is sufficient; a GPU is not required.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_data.py
python prepare_data.py
python train.py
streamlit run app.py
```

If PowerShell blocks activation, use:
```powershell
.venv\Scripts\activate.bat
```

## Setup (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
python prepare_data.py
python train.py
streamlit run app.py
```

## Dataset

Source: CIC-IDS2017, Canadian Institute for Cybersecurity. Download script retrieves four labelled CSV files from a public Hugging Face mirror of CIC-IDS2017. The original files are not bundled because they are large. Downloaded data is sampled by label into `data/processed_sample.csv`.

If the download mirror is unavailable, download the CSV flow files from the official dataset page:
https://www.unb.ca/cic/datasets/ids-2017.html
Place the CSVs in `data/raw/` and run `python prepare_data.py`.

Files used by the script:
- Tuesday-WorkingHours.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv

The download size can still be a few hundred MB. The *working sample* is capped at up to 2,500 records per label. Reduce `MAX_ROWS_PER_LABEL` in `src/config.py` to 500 or 1,000 for a smaller/lower-memory demo.

## Run with a chosen number of workers

```bash
python train.py --workers 2
python train.py --workers 4
```

Use a modest worker count. Do not set workers higher than available logical CPU threads. The script compares sequential and parallel evaluation of the same six candidate configurations.

## Dashboard pages

1. Security Overview — traffic decision chart and alerts.
2. Live CSV Analysis — upload compatible flow records and classify them.
3. Parallel Performance — execution time, speedup, and efficiency.
4. Model Evaluation — metrics, class report, and confusion matrix.
5. About & Viva — project explanation and viva preparation.

## Output files

- `artifacts/sentinelx_bundle.joblib` — trained RF, Isolation Forest, imputer and metadata.
- `reports/metrics.json` — measured metrics and parallel performance.
- `reports/test_predictions.csv` — known and withheld-class decisions.
- `data/processed_sample.csv` — sampled real dataset.

## Methodology and interpretation

- RF accuracy/F1 are evaluated on the known-class held-out split.
- Unknown PortScan recall is the fraction of withheld PortScan rows flagged as anomalous.
- BENIGN anomaly false-positive rate is measured on held-out BENIGN rows.
- Speedup = sequential execution time / parallel execution time.
- Efficiency = speedup / worker count × 100%.

A parallel run can be slower on small data because process creation and data transfer have overhead. Report the measured values, even if speedup is below 1.

## Important note on live upload

The uploaded CSV must contain compatible CICFlowMeter numeric feature columns. IP addresses, timestamps, flow identifiers, and any Label column are ignored. This is an offline CSV analyser, not a packet sniffer or a production network firewall.

## Dataset citation

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. ICISSP.
https://doi.org/10.5220/0006639801080116
