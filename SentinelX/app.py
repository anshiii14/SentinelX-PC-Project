import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

from src.pipeline import load_bundle, predict_frame
from src.config import METRICS_PATH, PREDICTIONS_PATH, PROCESSED_PATH

st.set_page_config(page_title="SentinelX | Network Defense", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(100,116,139,.08); padding: 14px; border-radius: 12px;}
.hero {padding: 22px 24px; border-radius: 16px; background: linear-gradient(110deg,#111827,#1e3a5f); color:white; margin-bottom:18px;}
.hero h1 {font-size:2.15rem; margin:0;}
.hero p {color:#d1d5db; margin:.35rem 0 0 0;}
.small-note {color: #64748b; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🛡️ SENTINELX</h1><p>Parallel Hybrid Network Intrusion Detection & Unknown Anomaly Intelligence</p></div>', unsafe_allow_html=True)

if not Path("artifacts/sentinelx_bundle.joblib").exists():
    st.error("Trained model not found. Run `python download_data.py`, `python prepare_data.py`, then `python train.py`.")
    st.stop()

bundle = load_bundle()
metrics = bundle["metrics"]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Open module", ["Security Overview", "Live CSV Analysis", "Parallel Performance", "Model Evaluation", "About & Viva"])
st.sidebar.caption(f"CPU workers configured: {bundle.get('workers', 1)}")

if page == "Security Overview":
    st.subheader("Security operations overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Training records", f"{metrics['known_train_rows']:,}")
    c2.metric("Known-class accuracy", f"{metrics['random_forest_accuracy']*100:.2f}%")
    c3.metric("Unknown PortScan recall", f"{metrics['unknown_portscan_detection_recall']*100:.2f}%")
    c4.metric("Parallel speedup", f"{metrics['speedup']:.2f}×")
    st.markdown("### Detection pipeline")
    a,b,c = st.columns(3)
    a.info("**Random Forest**\n\nRecognises attack classes learned during supervised training.")
    b.warning("**Isolation Forest**\n\nFlags traffic that deviates from normal BENIGN behaviour.")
    c.success("**Hybrid decision**\n\nCombines both outputs into an actionable traffic decision.")
    if Path(PREDICTIONS_PATH).exists():
        pred = pd.read_csv(PREDICTIONS_PATH)
        counts = pred["HybridDecision"].value_counts().rename_axis("Decision").reset_index(name="Flows")
        left,right = st.columns([1.2,1])
        with left:
            st.plotly_chart(px.pie(counts, names="Decision", values="Flows",
                                   hole=.48, title="Hybrid decisions in evaluation set"),
                            use_container_width=True)
        with right:
            st.plotly_chart(px.bar(counts, x="Decision", y="Flows", color="Decision",
                                   title="Traffic decision distribution"),
                            use_container_width=True)
        st.markdown("### Recent evaluation alerts")
        st.dataframe(pred[pred["HybridDecision"] != "BENIGN"].head(15),
                     use_container_width=True, hide_index=True)
    st.caption(metrics.get("note",""))

elif page == "Live CSV Analysis":
    st.subheader("Analyse an unseen traffic CSV")
    st.write("Upload CICFlowMeter-style flow records. The CSV should contain the same numeric flow features used during training. A Label column is optional and is ignored for prediction.")
    uploaded = st.file_uploader("Upload network-flow CSV", type=["csv"])
    if uploaded:
        frame = pd.read_csv(uploaded, low_memory=False)
        st.write(f"Uploaded **{len(frame):,}** flows and **{len(frame.columns)}** columns.")
        st.dataframe(frame.head(5), use_container_width=True)
        if st.button("Run parallel hybrid detection", type="primary"):
            with st.spinner("Scoring traffic batches across CPU workers..."):
                result = predict_frame(frame, bundle)
            merged = pd.concat([frame.reset_index(drop=True), result], axis=1)
            st.session_state["latest_result"] = merged
            st.success("Analysis complete.")
    if "latest_result" in st.session_state:
        result = st.session_state["latest_result"]
        counts = result["FinalDecision"].value_counts().rename_axis("Decision").reset_index(name="Flows")
        x,y,z = st.columns(3)
        x.metric("Flows analysed", f"{len(result):,}")
        y.metric("Suspicious flows", f"{(result.FinalDecision != 'BENIGN').sum():,}")
        z.metric("Unknown anomalies", f"{(result.FinalDecision == 'Unknown Anomaly').sum():,}")
        st.plotly_chart(px.bar(counts, x="Decision", y="Flows", color="Decision",
                               title="Detection summary"), use_container_width=True)
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button("Download detection report CSV",
                           result.to_csv(index=False).encode("utf-8"),
                           "sentinelx_detection_report.csv", "text/csv")

elif page == "Parallel Performance":
    st.subheader("Parallel computing performance")
    seq = metrics["sequential_search_seconds"]
    par = metrics["parallel_search_seconds"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Sequential runtime", f"{seq:.2f} s")
    c2.metric("Parallel runtime", f"{par:.2f} s")
    c3.metric("Speedup", f"{metrics['speedup']:.2f}×")
    c4.metric("Efficiency", f"{metrics['parallel_efficiency_percent']:.1f}%")
    perf = pd.DataFrame({"Mode":["Sequential","Parallel"],
                         "Execution time (seconds)":[seq,par]})
    st.plotly_chart(px.bar(perf, x="Mode", y="Execution time (seconds)",
                           color="Mode", title="Independent model-search runtime"),
                    use_container_width=True)
    st.markdown("#### Formulas")
    st.latex(r"Speedup = \\frac{T_{sequential}}{T_{parallel}}")
    st.latex(r"Efficiency = \\frac{Speedup}{Number\\ of\\ Workers}\\times 100")
    st.markdown("#### What is parallelised?")
    st.write("Six independent Random Forest configurations are evaluated concurrently using joblib's process-based `loky` backend. Traffic scoring can also be split into independent batches and processed by multiple workers.")
    st.warning("Speedup is workload- and hardware-dependent. Small datasets may show limited speedup because process startup and data-transfer overhead can exceed the computation saved.")

elif page == "Model Evaluation":
    st.subheader("Model evaluation")
    st.write("Random Forest metrics are measured on the held-out known-class test split. PortScan is kept out of supervised training and evaluated separately as an unseen category.")
    st.metric("Random Forest macro F1", f"{metrics['random_forest_macro_f1']:.4f}")
    st.metric("PortScan unknown-detection recall", f"{metrics['unknown_portscan_detection_recall']:.4f}")
    st.metric("BENIGN anomaly false-positive rate", f"{metrics['benign_anomaly_false_positive_rate']:.4f}")
    report = metrics["classification_report"]
    rows = []
    for label, vals in report.items():
        if isinstance(vals, dict) and "f1-score" in vals:
            rows.append({"Class":label, "Precision":vals["precision"],
                         "Recall":vals["recall"], "F1-score":vals["f1-score"],
                         "Support":vals["support"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    cm = pd.DataFrame(metrics["confusion_matrix"],
                      index=metrics["confusion_matrix_labels"],
                      columns=metrics["confusion_matrix_labels"])
    st.plotly_chart(px.imshow(cm, text_auto=True, aspect="auto",
                              title="Random Forest confusion matrix",
                              labels=dict(x="Predicted",y="Actual",color="Flows")),
                    use_container_width=True)
    st.caption("Unknown anomaly means the traffic is statistically unusual relative to learned benign traffic. It does not identify the exact attack family.")

else:
    st.subheader("Project explanation")
    st.markdown("""
**Problem:** Signature-based detection can miss previously unseen attack behaviour and analysing high-volume traffic can be computationally expensive.

**SentinelX solution:**
- Random Forest classifies known traffic categories.
- Isolation Forest is fitted only on BENIGN flows and identifies outliers.
- The hybrid decision engine labels recognised attacks, benign traffic, and unknown anomalies.
- Joblib multiprocessing evaluates independent model configurations and parallelises traffic-batch inference.

**Novelty experiment:** PortScan is intentionally excluded from supervised training. It is then passed through the hybrid detector as an unseen attack family.

**Limitations:** The experiment is a controlled offline benchmark using CIC-IDS2017 flow records. An anomaly is a suspicious signal, not proof of malicious intent. Real deployment requires live flow collection, threshold calibration, drift monitoring, and security validation.
    """)
    st.markdown("### Viva questions")
    qa = [
        ("Why Random Forest?", "It is an ensemble of decision trees, handles nonlinear feature interactions, is robust on tabular data, and supports feature importance."),
        ("Why Isolation Forest?", "It isolates unusual observations using random partitions, so it can flag outliers without needing attack labels."),
        ("How do you test unknown attacks?", "We withhold all PortScan-labelled rows from supervised training and evaluate them separately."),
        ("What is speedup?", "Sequential runtime divided by parallel runtime."),
        ("What is parallel efficiency?", "Speedup divided by the number of workers, multiplied by 100."),
        ("Why not require a GPU?", "The selected scikit-learn models and independent experiment jobs are CPU-oriented; multicore CPU processing demonstrates the parallel-computing objective."),
        ("Does Unknown Anomaly mean PortScan?", "No. The model only reports unusual behaviour. PortScan is the known ground-truth category used to evaluate novelty detection.")
    ]
    for q,a in qa:
        with st.expander(q):
            st.write(a)
