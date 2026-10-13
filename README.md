# AI-Based Intrusion Detection System (AI-IDS)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.1-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%201.5+-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS2017-green.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Accuracy](https://img.shields.io/badge/Benchmark%20Accuracy-96.39%25-brightgreen.svg)](#machine-learning-architecture)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)]()

An enterprise-grade, machine-learning-powered **Network Intrusion Detection System (NIDS)** and **Security Operations Center (SOC) Console**.

The system ingests raw network flow captures (PCAP/CSV), extracts 78 statistical traffic features, and performs real-time multi-class classification across 15 distinct cyberattack categories using an optimized Random Forest ensemble model.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Raw Network Traffic (CSV)       |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     Input Validation & Preprocessing   |
                      |  - 78 Feature Alignment Verification  |
                      |  - Missing / Infinite Value Cleaning  |
                      |  - StandardScaler Normalization       |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       Machine Learning Engine         |
                      |    RandomForestClassifier (100 Trees) |
                      |    15 Attack Class Multi-Classifier   |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       SOC Analyst Web Interface       |
                      |  - Real-Time Traffic Composition      |
                      |  - Multi-Class Threat Breakdown       |
                      |  - Incident Escalation & Response     |
                      |  - Auditable Intelligence Exports     |
                      +---------------------------------------+
```

---

## Machine Learning Architecture

* **Model Family:** `RandomForestClassifier` (100 Decision Tree estimators, balanced sampling, parallelized inference)
* **Feature Dimension:** 78 numerical network flow features (Flow Duration, Packet Lengths, Inter-Arrival Times, TCP Flags, Window Sizes, Subflow Rates)
* **Feature Preprocessing:** `StandardScaler` trained on the benchmark CIC-IDS2017 dataset
* **Classification Scope:** 15 multi-class target labels:
  1. `BENIGN` (Normal Traffic)
  2. `DDoS`
  3. `PortScan`
  4. `Bot`
  5. `Infiltration`
  6. `DoS Hulk`
  7. `DoS GoldenEye`
  8. `DoS slowloris`
  9. `DoS Slowhttptest`
  10. `FTP-Patator` (Brute Force)
  11. `SSH-Patator` (Brute Force)
  12. `Web Attack - Brute Force`
  13. `Web Attack - XSS`
  14. `Web Attack - Sql Injection`
  15. `Heartbleed`
* **Benchmark Performance:** **96.39% accuracy** on the CIC-IDS2017 test evaluation set.
* **Inference Throughput:** ~3,100+ network flows/second on CPU execution.

---

## Application Features

* **SOC Dashboard:** Real-time metrics tracking total flows, benign traffic, detected threats, attack distributions, and detection pipeline health.
* **Dynamic Network Composition:** Horizontal bar visualization contrasting malicious vs. benign network volume.
* **Traffic Analysis Engine:** Drag-and-drop CSV ingestion with asynchronous job polling and validation against required feature sets.
* **Threat Intelligence Table:** Per-flow forensic triage with severity badges and rule-based mitigation recommendations (e.g., firewall filtering, IP blocking, host isolation).
* **Incident Escalation:** Threat-to-incident escalation workflow with full status lifecycle management (`ACTIVE` $\rightarrow$ `INVESTIGATING` $\rightarrow$ `RESOLVED`).
* **Global Search:** Top navigation search bar to locate threats, risk tiers, record IDs, and incident dossiers.
* **Forensic Export:** One-click CSV and JSON exports of all classified network flow events.

---

## Repository Structure

```text
├── app.py                     # Flask application routes, APIs & security gates
├── wsgi.py                    # Production WSGI entry point (Gunicorn / Waitress)
├── ids_model_package.pkl      # Bundled model, scaler, features & class dictionary
├── feature_names.npy          # 78 network flow feature names array
├── class_mapping.json         # Numeric-to-label class mapping
├── requirements.txt           # Pinned production Python dependencies
├── render.yaml                # Render Blueprint infrastructure-as-code
├── .env.example               # Environment variables template
├── .gitignore                 # Production Git exclusion rules
├── sample_data/               # Non-sensitive synthetic sample for demonstration
│   └── synthetic_traffic_sample.csv
├── src/                       # Backend ML prediction modules
│   ├── __init__.py
│   ├── predict.py
│   └── test_real_prediction.py
├── static/                    # CSS, JavaScript & chart components
├── templates/                 # SOC console HTML templates (Tailwind & Lucide)
├── results/                   # Benchmark evaluation figures & metrics
└── notebooks/                 # Academic exploration & model training notebooks
```

---

## Local Installation & Setup

### Prerequisites
* Python 3.11 or Python 3.12
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/Anushka-Jadhav/AI-Based-Intrusion-Detection-System.git
cd AI-Based-Intrusion-Detection-System
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```
*(Default demo credentials: `ADMIN_USERNAME=admin`, `ADMIN_PASSWORD=admin`)*

### 5. Launch the Application
```bash
# Development server
python app.py

# Production WSGI server (Linux/macOS)
gunicorn --workers 1 --bind 0.0.0.0:5000 wsgi:application

# Production WSGI server (Windows)
python wsgi.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## Deployment on Render

This project is configured for one-click deployment on [Render](https://render.com/).

### Option A: Using the Render Blueprint (`render.yaml`)
1. Create a free account at [Render](https://render.com/).
2. Click **New +** $\rightarrow$ **Blueprint**.
3. Connect your GitHub repository.
4. Render will read `render.yaml` and configure the web service automatically.

### Option B: Manual Web Service Setup
* **Environment:** Python
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `gunicorn --workers 1 --bind 0.0.0.0:$PORT wsgi:application`
* **Environment Variables:**
  * `SECRET_KEY`: *(Auto-generate or enter a secure string)*
  * `ADMIN_USERNAME`: `admin`
  * `ADMIN_PASSWORD`: `admin123`
  * `FLASK_DEBUG`: `false`

> **Memory Optimization Note:** The Gunicorn server is configured with `--workers 1`. This maintains memory usage at ~260 MB, well within Render's free tier 512 MB RAM limit while providing robust inference throughput.

---

## Testing & Demonstration

A synthetic, non-sensitive sample dataset is included in `sample_data/synthetic_traffic_sample.csv` (100 flow records with all 78 required feature columns).

To test:
1. Log in to the console (`admin` / `admin`).
2. Navigate to **Traffic Analysis**.
3. Upload `sample_data/synthetic_traffic_sample.csv`.
4. Observe the batch inference progress bar reach 100%.
5. View the updated **Dashboard**, **Threats**, and **Reports** pages with live classification outputs.

---

## Academic Project Information

* **Degree:** Bachelor of Science in Computer Science (B.Sc. CS)
* **Author:** Anushka Jadhav
* **Dataset Reference:** Canadian Institute for Cybersecurity (CIC-IDS2017)
* **Primary Classifier:** Scikit-Learn Random Forest Ensemble
