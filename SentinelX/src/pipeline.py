import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import parallel_backend

from src.config import (
    PROCESSED_PATH, MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH,
    ARTIFACT_DIR, REPORT_DIR, UNKNOWN_LABEL_KEYWORDS, RANDOM_STATE
)

DROP_COLUMNS = {
    "Label", "Flow ID", "Source IP", "Destination IP", "Timestamp",
    "Src IP", "Dst IP"
}

def canonical_label(label):
    s = str(label).strip().lower()
    if s == "benign":
        return "BENIGN"
    if "portscan" in s:
        return "PortScan"
    if "ddos" in s:
        return "DDoS"
    if s.startswith("dos") or "dos " in s:
        return "DoS"
    if "patator" in s or "brute" in s:
        return "Brute Force"
    if "web attack" in s or "sql injection" in s or "xss" in s:
        return "Web Attack"
    if "bot" in s:
        return "Bot"
    if "infiltration" in s:
        return "Infiltration"
    if "heartbleed" in s:
        return "Heartbleed"
    return str(label).strip()

def prepare_features(frame, feature_columns=None):
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # CIC-IDS2017 has numeric flow features. IPs, timestamps and IDs are
    # excluded to avoid memorising endpoints or capture-specific identifiers.
    drop = [c for c in df.columns if c in DROP_COLUMNS or c.lower() in
            {"flow id", "source ip", "destination ip", "timestamp"}]
    if "Label" in df.columns:
        drop.append("Label")
    X = df.drop(columns=list(set(drop)), errors="ignore")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns)
    return X

def _fit_candidate(X, y, n_estimators, max_depth, min_samples_leaf, seed):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=1, random_state=seed
    )
    model.fit(X, y)
    pred = model.predict(X)
    return {
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf
        },
        "train_macro_f1": float(f1_score(y, pred, average="macro"))
    }


def train_system(workers=None):
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError("Sample dataset missing. Run python prepare_data.py first.")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "Label" not in df:
        raise ValueError("Dataset must contain a Label column.")
    df["OriginalLabel"] = df["Label"].astype(str).str.strip()
    df["Class"] = df["OriginalLabel"].map(canonical_label)

    # PortScan is completely withheld from supervised training and validation.
    unknown_mask = df["Class"].str.lower().str.contains("|".join(UNKNOWN_LABEL_KEYWORDS), na=False)
    unknown_df = df[unknown_mask].copy()
    known_df = df[~unknown_mask].copy()

    if "BENIGN" not in set(known_df["Class"]):
        raise ValueError("No BENIGN rows found. Include source CSVs containing BENIGN.")
    if unknown_df.empty:
        raise ValueError("No PortScan rows found. Include Friday-Afternoon-PortScan CSV.")

    X_all = prepare_features(known_df)
    feature_columns = [c for c in X_all.columns if X_all[c].notna().any()]
    X_all = X_all[feature_columns]
    y_all = known_df["Class"].astype(str)

    # 60/20/20 train/validation/test split. Validation selects the RF configuration
    # and calibrates the anomaly threshold; test is held back for final reporting.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE,
        stratify=y_trainval
    )

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_val_i = imputer.transform(X_val)
    X_test_i = imputer.transform(X_test)

    cpu_count = os.cpu_count() or 2
    workers = max(1, min(int(workers or max(1, cpu_count - 1)), cpu_count))
    configs = [
        (100, None, 1), (150, None, 1), (200, 20, 1),
        (200, None, 2), (250, 25, 2), (300, None, 1)
    ]

    def evaluate_candidate(Xtr, ytr, Xvalid, yvalid, n_estimators,
                          max_depth, min_samples_leaf, seed):
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample", n_jobs=1, random_state=seed
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xvalid)
        return {
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf
            },
            "validation_macro_f1": float(f1_score(yvalid, pred, average="macro"))
        }

    t0 = time.perf_counter()
    sequential_results = [
        evaluate_candidate(X_train_i, y_train, X_val_i, y_val, *cfg, RANDOM_STATE + i)
        for i, cfg in enumerate(configs)
    ]
    sequential_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel_results = Parallel(n_jobs=workers, backend="loky")(
        delayed(evaluate_candidate)(
            X_train_i, y_train, X_val_i, y_val, *cfg, RANDOM_STATE + i
        ) for i, cfg in enumerate(configs)
    )
    parallel_seconds = time.perf_counter() - t0

    best_idx = int(np.argmax([r["validation_macro_f1"] for r in parallel_results]))
    best = parallel_results[best_idx]["params"]

    # Refit the selected classifier on train + validation; test remains untouched.
    X_fit = pd.concat([X_train, X_val], axis=0)
    y_fit = pd.concat([y_train, y_val], axis=0)
    # Keep one train-fitted imputer shared by both models; validation/test
    # rows never influence preprocessing statistics.
    X_fit_i = imputer.transform(X_fit)
    X_test_final = X_test_i
    rf = RandomForestClassifier(
        **best, class_weight="balanced_subsample",
        n_jobs=workers, random_state=RANDOM_STATE
    )
    rf.fit(X_fit_i, y_fit)
    known_pred = rf.predict(X_test_final)

    # Isolation Forest learns only from BENIGN training+validation flows.
    benign_idx = np.where(np.asarray(y_train) == "BENIGN")[0]
    if len(benign_idx) < 20:
        raise ValueError("Not enough BENIGN training records for Isolation Forest.")
    iso = IsolationForest(
        n_estimators=200, contamination=0.03,
        random_state=RANDOM_STATE, n_jobs=workers
    )
    iso.fit(X_train_i[benign_idx])

    # Calibrate anomaly threshold on benign validation flows only.
    val_benign_idx = np.where(np.asarray(y_val) == "BENIGN")[0]
    val_benign_scores = -iso.decision_function(X_val_i[val_benign_idx])
    anomaly_threshold = float(np.quantile(val_benign_scores, 0.97))

    known_anomaly_scores = -iso.decision_function(X_test_final)
    known_is_anomaly = known_anomaly_scores > anomaly_threshold
    hybrid_known = [
        pred if pred != "BENIGN" else (
            "Unknown Anomaly" if anomaly else "BENIGN"
        )
        for pred, anomaly in zip(known_pred, known_is_anomaly)
    ]

    # Evaluate withheld PortScan. No PortScan row is used in RF or IF fitting.
    X_unknown = prepare_features(unknown_df, feature_columns)
    X_unknown_i = imputer.transform(X_unknown)
    unknown_scores = -iso.decision_function(X_unknown_i)
    unknown_flags = unknown_scores > anomaly_threshold
    unknown_rf_pred = rf.predict(X_unknown_i)
    unknown_hybrid = np.where(unknown_flags, "Unknown Anomaly", unknown_rf_pred)

    unknown_recall = float(np.mean(unknown_flags))
    test_benign_idx = np.where(np.asarray(y_test) == "BENIGN")[0]
    benign_fp = float(np.mean(known_is_anomaly[test_benign_idx])) if len(test_benign_idx) else 0.0

    report = classification_report(y_test, known_pred, output_dict=True, zero_division=0)
    cm_labels = sorted(set(y_test) | set(known_pred))
    cm = confusion_matrix(y_test, known_pred, labels=cm_labels)

    metrics = {
        "dataset_rows": int(len(df)),
        "known_train_rows": int(len(X_fit)),
        "known_validation_rows": int(len(X_val)),
        "known_test_rows": int(len(X_test)),
        "unknown_portscan_rows": int(len(unknown_df)),
        "known_classes": sorted(y_all.unique().tolist()),
        "withheld_class": "PortScan",
        "workers": workers,
        "cpu_logical_count": cpu_count,
        "sequential_search_seconds": sequential_seconds,
        "parallel_search_seconds": parallel_seconds,
        "speedup": float(sequential_seconds / parallel_seconds) if parallel_seconds else None,
        "parallel_efficiency_percent": float((sequential_seconds / parallel_seconds) / workers * 100) if parallel_seconds else None,
        "best_parameters": best,
        "candidate_validation_results": parallel_results,
        "random_forest_accuracy": float(accuracy_score(y_test, known_pred)),
        "random_forest_macro_f1": float(f1_score(y_test, known_pred, average="macro")),
        "unknown_portscan_detection_recall": unknown_recall,
        "benign_anomaly_false_positive_rate": benign_fp,
        "classification_report": report,
        "confusion_matrix_labels": cm_labels,
        "confusion_matrix": cm.tolist(),
        "parallel_batch_scoring_workers": workers,
        "note": "PortScan is withheld from supervised training. Unknown Anomaly is a novelty alert, not a confirmed attack identity."
    }

    bundle = {
        "rf": rf, "isolation_forest": iso, "imputer": imputer,
        "feature_columns": feature_columns,
        "anomaly_threshold": anomaly_threshold,
        "workers": workers,
        "metrics": metrics
    }
    joblib.dump(bundle, MODEL_PATH)

    predictions = pd.DataFrame({
        "Actual": list(y_test.astype(str)),
        "RandomForest": list(known_pred),
        "AnomalyScore": list(known_anomaly_scores),
        "HybridDecision": list(hybrid_known)
    })
    unknown_out = pd.DataFrame({
        "Actual": ["UNSEEN: PortScan"] * len(unknown_df),
        "RandomForest": list(unknown_rf_pred),
        "AnomalyScore": list(unknown_scores),
        "HybridDecision": list(unknown_hybrid)
    })
    pd.concat([predictions, unknown_out], ignore_index=True).to_csv(PREDICTIONS_PATH, index=False)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\\n===== SENTINELX TRAINING COMPLETE =====")
    print(f"Known classes: {metrics['known_classes']}")
    print(f"Withheld class: PortScan ({len(unknown_df):,} rows)")
    print(f"RF accuracy: {metrics['random_forest_accuracy']:.4f}")
    print(f"Unknown PortScan detection recall: {unknown_recall:.4f}")
    print(f"BENIGN anomaly false-positive rate: {benign_fp:.4f}")
    print(f"Sequential search: {sequential_seconds:.2f}s")
    print(f"Parallel search ({workers} workers): {parallel_seconds:.2f}s")
    print(f"Speedup: {metrics['speedup']:.2f}x")
    print(f"Parallel efficiency: {metrics['parallel_efficiency_percent']:.1f}%")
    print(f"Saved model: {MODEL_PATH}")
    return metrics

def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python train.py")
    return joblib.load(MODEL_PATH)

def predict_frame(frame, bundle=None, workers=None):
    bundle = bundle or load_bundle()
    feature_columns = bundle["feature_columns"]
    X = prepare_features(frame, feature_columns)
    X_i = bundle["imputer"].transform(X)
    rf = bundle["rf"]
    iso = bundle["isolation_forest"]
    threshold = bundle["anomaly_threshold"]
    workers = max(1, int(workers or bundle.get("workers", 1)))

    # Parallelise independent traffic batches, then preserve original row order.
    chunks = np.array_split(np.arange(len(X_i)), min(workers, max(1, len(X_i))))
    def score_chunk(indices):
        if len(indices) == 0:
            return indices, np.array([]), np.array([]), np.array([])
        part = X_i[indices]
        return indices, rf.predict(part), -iso.decision_function(part), iso.predict(part)
    results = Parallel(n_jobs=workers, backend="loky")(
        delayed(score_chunk)(idx) for idx in chunks if len(idx)
    )
    preds = np.empty(len(X_i), dtype=object)
    scores = np.empty(len(X_i), dtype=float)
    flags = np.empty(len(X_i), dtype=int)
    for idx, p, s, f in results:
        preds[idx], scores[idx], flags[idx] = p, s, f

    decisions = []
    for pred, score in zip(preds, scores):
        if pred != "BENIGN":
            decisions.append(pred)
        elif score > threshold:
            decisions.append("Unknown Anomaly")
        else:
            decisions.append("BENIGN")

    output = pd.DataFrame({
        "RandomForestPrediction": preds,
        "AnomalyScore": scores,
        "IsolationForestFlag": np.where(scores > threshold, "Anomaly", "Normal"),
        "FinalDecision": decisions
    })
    return output
