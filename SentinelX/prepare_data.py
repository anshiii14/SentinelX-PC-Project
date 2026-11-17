import re
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import RAW_DIR, PROCESSED_PATH, MAX_ROWS_PER_LABEL, RANDOM_STATE

LABEL_COL = "Label"

def clean_label(value):
    return str(value).strip()

def main():
    paths = sorted(RAW_DIR.glob("*.csv"))
    if not paths:
        raise SystemExit("No CIC-IDS2017 CSVs found. Run: python download_data.py")

    samples = []
    print("Reading real CIC-IDS2017 files in chunks and sampling by label...")
    for path in paths:
        print(f"\n[scan] {path.name}")
        kept = {}
        try:
            reader = pd.read_csv(path, chunksize=100_000, low_memory=False, encoding_errors="replace")
            for chunk in reader:
                chunk.columns = [str(c).strip() for c in chunk.columns]
                if LABEL_COL not in chunk.columns:
                    raise ValueError(f"'Label' column not found in {path.name}")
                chunk[LABEL_COL] = chunk[LABEL_COL].map(clean_label)
                for label, group in chunk.groupby(LABEL_COL):
                    current = kept.get(label, [])
                    remaining = MAX_ROWS_PER_LABEL - sum(len(x) for x in current)
                    if remaining > 0:
                        take = group.sample(n=min(remaining, len(group)),
                                            random_state=RANDOM_STATE)
                        current.append(take)
                        kept[label] = current
            for label, parts in kept.items():
                if parts:
                    samples.append(pd.concat(parts, ignore_index=True))
                    print(f"  {label}: {sum(len(x) for x in parts):,} sampled")
        except Exception as exc:
            print(f"[warning] Could not process {path.name}: {exc}")
            raise

    if not samples:
        raise SystemExit("No records were sampled. Check downloaded CSV files.")
    data = pd.concat(samples, ignore_index=True)
    data.columns = [str(c).strip() for c in data.columns]
    data[LABEL_COL] = data[LABEL_COL].map(clean_label)

    # Remove duplicate rows and CICFlowMeter invalid values.
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.drop_duplicates()
    data = data.dropna(subset=[LABEL_COL])

    # Keep the first representative sample for each label across source files.
    balanced_parts = []
    for _, group in data.groupby(LABEL_COL):
        balanced_parts.append(group.sample(
            n=min(MAX_ROWS_PER_LABEL, len(group)), random_state=RANDOM_STATE
        ))
    data = pd.concat(balanced_parts, ignore_index=True)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved {len(data):,} real flow records to {PROCESSED_PATH}")
    print("\nClass distribution:")
    print(data[LABEL_COL].value_counts().to_string())
    print("\nImportant: PortScan is reserved as the unseen attack category in training.")

if __name__ == "__main__":
    main()
