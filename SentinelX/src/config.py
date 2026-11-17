from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
PROCESSED_PATH = ROOT / "data" / "processed_sample.csv"
MODEL_PATH = ARTIFACT_DIR / "sentinelx_bundle.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"

# PortScan is deliberately withheld from supervised training for the
# unknown-attack experiment. This is a controlled novelty-detection test.
UNKNOWN_LABEL_KEYWORDS = ("portscan",)
MAX_ROWS_PER_LABEL = 2500
RANDOM_STATE = 42
