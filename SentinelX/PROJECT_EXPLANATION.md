# SentinelX — presentation notes

## 60-second explanation

SentinelX is a hybrid Network Intrusion Detection System. It uses Random Forest to classify known network attacks and Isolation Forest to flag traffic that behaves differently from normal network activity. To demonstrate detection of an unseen attack, PortScan is excluded from supervised training and used only during final evaluation. Parallel computing is implemented using multiple CPU processes to evaluate independent Random Forest configurations and score traffic batches. The dashboard shows predictions, alerts, classification metrics, and measured sequential-versus-parallel runtime.

## What makes the project parallel?

1. Six independent Random Forest configurations are evaluated.
2. Sequential mode runs them one after another.
3. Parallel mode dispatches them to separate worker processes with joblib (`loky`).
4. The best configuration is selected using macro-F1 on training data (the final reported RF metrics are on the held-out test split).
5. Prediction requests are divided into independent batches and scored concurrently.

## Unknown anomaly experiment

- PortScan records are filtered out before supervised training.
- Isolation Forest is fitted on BENIGN training rows only.
- Anomaly threshold is calibrated from held-out BENIGN traffic.
- PortScan records are passed to the detector only after training.
- Recall measures what fraction of withheld PortScan traffic is flagged.

Do not say that the system recognises the exact unknown attack. It detects that the flow is unusual.

## Demo sequence

1. Open Security Overview.
2. Show the held-out PortScan experiment and evaluation metrics.
3. Open Parallel Performance and explain speedup/efficiency.
4. Open Model Evaluation and explain confusion matrix and unknown recall.
5. Open Live CSV Analysis and upload a compatible flow CSV.
6. Download the resulting detection report.

## Viva questions

**Q. What is an IDS?**  
A system that monitors network or system activity for suspicious or malicious behaviour.

**Q. Why hybrid detection?**  
Supervised classification recognises known labels; unsupervised anomaly detection can flag unusual observations outside those known labels.

**Q. Why Random Forest?**  
It combines many decision trees, handles nonlinear relationships in tabular data, and is relatively robust.

**Q. Why Isolation Forest?**  
It identifies outliers by isolating observations through random feature and split selection.

**Q. How is unknown attack detection validated?**  
A complete labelled attack family (PortScan) is withheld from supervised training and then tested separately.

**Q. What does an anomaly alert mean?**  
It means the flow differs from learned benign behaviour; it is not proof of a particular attack.

**Q. What is speedup?**  
Sequential execution time divided by parallel execution time.

**Q. What is efficiency?**  
Speedup divided by the number of workers, multiplied by 100.

**Q. Why CPU instead of GPU?**  
The chosen scikit-learn algorithms and process-based model-search tasks are CPU-oriented. A GPU is not necessary to demonstrate multicore parallelism.

**Q. Why can parallel be slower?**  
For small workloads, process startup, serialization, and inter-process communication can cost more than the computation saved.

## Limitations to acknowledge

- CIC-IDS2017 is a benchmark dataset, not live traffic from your campus network.
- PortScan is the chosen unseen category in this experiment; it does not prove generalisation to every possible zero-day attack.
- Isolation Forest threshold needs calibration to the deployment environment.
- A practical production IDS needs live flow collection, monitoring, drift handling, and analyst review.
