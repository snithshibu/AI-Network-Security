# 🛡️ AI-Powered Network Security System

This project is submitted as part of the **Intel® Unnati Internship Program 2025** under the theme  
**AI/ML for Networking – Network Security**.

It integrates two machine learning components into a modern web application to enhance automated network defense:

1. 🚨 **Threat Detection**  
   Detects and classifies potentially malicious URLs into types: `SQL Injection`, `XSS`, or `Benign`.

2. 🚦 **Traffic Classification**  
   Classifies network traffic flows into application types like HTTP, SSL, GOOGLE, etc., using flow-level metadata.

Both systems are unified within a clean, interactive **Streamlit dashboard** that supports real-time and batch analysis.

---

## 🎯 Project Deliverables

| Deliverable | Description | Status |
|-------------|-------------|--------|
| ✅ AI-Powered Threat Detection | Detect and classify SQLi/XSS from URLs | ✔️ Multi-class model trained |
| ✅ Traffic Classification Model | App ID detection from flow-level metadata | ✔️ Implemented via Random Forest |
| ✅ Real-Time + Batch Support | Accepts both single inputs and CSV uploads | ✔️ Supported in Streamlit UI |
| ✅ Privacy-Preserving Analysis | No decryption or DPI; metadata-based | ✔️ Fully compliant |
| ✅ Downloadable Results | Predictions can be exported to CSV | ✔️ Enabled |
| ✅ Modular & Scalable Design | Models are decoupled and retrainable | ✔️ Structured for future extensions |

---

## 🧩 Key Features

### 🔍 Threat Detection (Multi-Class)
- Input: Single URL or CSV of URLs
- Output Classes:
  - ⚠️ `SQL_INJECTION`
  - ⚠️ `XSS`
  - ✅ `Benign`
- Feature engineering includes length, entropy, tokens, symbols, and special patterns

### 🚦 Traffic Classification
- Input: CSV with 87 flow-based metadata features
- Output Classes: HTTP, SSL, GOOGLE, etc.
- Scalable to more classes with retraining

---

## 🛠️ Tech Stack

- **Python 3.12**
- **Streamlit** – Web dashboard
- **scikit-learn** – Machine learning models
- **pandas**, **numpy** – Data handling
- **joblib** – Model serialization
- **StandardScaler**, **LabelEncoder** – Preprocessing tools

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
python -m venv env <br>
env\Scripts\activate        # On Windows <br>
#### OR <br>
source env/bin/activate     # On Linux/macOS

### Step 2: Install dependencies
pip install -r requirements.txt

### Step 3: Launch the app
streamlit run app.py

---

## 📈 Sample Outputs

| URL                                          | Prediction                    |
| -------------------------------------------- | ----------------------------- |
| `http://abc.com?q=<script>alert(1)</script>` | ⚠️ Malicious — XSS            |
| `http://example.com?id=1 OR 1=1`             | ⚠️ Malicious — SQL\_INJECTION |
| `https://shop.com/product?item=123`          | ✅ Benign                      |

| Traffic Flow Metadata           | Prediction |
| ------------------------------- | ---------- |
| Flow with SSL protocol features | ✅ SSL      |
| Flow with GOOGLE app pattern    | ✅ GOOGLE   |


---

👨‍💻 Developed By <br>
Snith Shibu <br>
Sidharth Sumitra Gireesh <br>
Devananda S.R. <br>
Intel® Unnati Internship 2025 <br>
