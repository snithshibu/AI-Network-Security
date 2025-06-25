# 🛡️ AI-Powered Network Security System

This project is a submission for the **Intel® Unnati Internship Project** under the theme **AI/ML for Networking - Network Security**.  
It includes two core machine learning components:

1. 🚨 **Threat Detection**: Detects SQL Injection (SQLi) and Cross-Site Scripting (XSS) in URLs
2. 🚦 **Traffic Classification**: Classifies network flows into app types like HTTP, SSL, GOOGLE, etc.

Both systems are integrated into a modern **Streamlit web dashboard** for real-time and batch analysis.

---

## 🎯 Project Deliverables

| Deliverable | Status |
|------------|--------|
| ✅ Threat Detection using AI (SQLi/XSS) | ✔️ Implemented via Random Forest |
| ✅ Traffic Classification (App ID detection) | ✔️ Implemented via ML on flow metadata |
| ✅ Real-time + Batch Mode | ✔️ Streamlit UI with CSV upload support |
| ✅ Privacy-preserving analysis | ✔️ Only uses metadata, not decrypted payloads |
| ✅ Downloadable results | ✔️ Predictions can be exported as CSV |
| ✅ Modular and Scalable | ✔️ Models are decoupled and retrainable |

---

## 🧩 Features

### 🔍 Threat Detection
- Accepts a single URL or CSV of URLs
- Extracts features like entropy, token histograms, suspicious patterns
- Predicts: `Malicious` or `Benign`

### 🚦 Traffic Classification
- Accepts CSVs of network flow data with 87 metadata features
- Classifies traffic into app types (HTTP, SSL, GOOGLE, etc.)
- Scalable to more classes with retraining

---

## 🛠️ Tech Stack

- **Python 3.12**
- **Streamlit** – Frontend dashboard
- **scikit-learn** – Machine Learning models
- **joblib** – Model persistence
- **pandas**, **numpy** – Data handling
- **StandardScaler**, **LabelEncoder** – Preprocessing

---

## 📁 Folder Structure

AI-Network-security/ <br>
│ <br>
├── app.py # Streamlit UI <br>
├── requirements.txt # Dependencies <br>
├── sample_http.csv # URL dataset for threat detection <br>
├── data.csv # Flow-based dataset for classification <br>
│<br>
├── model/<br>
│ ├── rf_model.pkl<br>
│ ├── traffic_classifier.pkl<br>
│ ├── traffic_scaler.pkl<br>
│ └── traffic_label_encoder.pkl<br>
│<br>
├── utils/<br>
│ └── feature_extraction.py # Custom URL feature logic<br>
│<br>
├── notebooks/ # Jupyter notebooks for training<br>
│ ├── traffic_classification.ipynb<br>
│ └── explore_traffic_data.ipynb<br>

---

## ▶️ How to Run

bash
### Step 1: Create virtual environment
python -m venv env
env\Scripts\activate      # or source env/bin/activate (Linux/macOS)

### Step 2: Install dependencies
pip install -r requirements.txt

### Step 3: Launch the app
streamlit run app.py

---

## 📈 Sample Outputs

| Input Type | Output
|------------|--------|
http://example.com?query=<script>alert(1)</script> |	⚠️ Malicious
Flow with ProtocolName: SSL, 87 metadata fields |	✅ SSL

---

👨‍💻 Developed By <br>
Snith Shibu, Sidharth Sumitra Gireesh, Devananda S.R. <br>
Intel® Unnati Internship 2025 <br>
