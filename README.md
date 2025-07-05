# 🛡️ AI-Powered Network Security System

This project is submitted as part of the **Intel® Unnati Internship Trainin Program 2025** for the problem statement - **AI/ML for Networking – Network Security**.

It leverages **Machine Learning** to provide real-time **Threat Detection (XSS, SQL Injection, Benign)** and **Traffic Classification (App Protocols)** based on flow metadata.

1. 🚨 **Threat Detection**  
- Predicts if a given URL is:
  - ⚠️ XSS (Cross-site scripting)
  - 🧨 SQL Injection
  - ✅ Benign
- Displays a **confidence score** for predictions.
- Supports **single URL** or **CSV upload**.
- Built on hand-crafted feature extraction and a Random Forest classifier.

2. 🚦 **Traffic Classification**  
- Classifies network flow metadata (87 features) into:
  - `HTTP`, `SSL`, `GOOGLE`, `HTTP_CONNECT`, `HTTP_PROXY`
- Uses scikit-learn pipelines with `LabelEncoder`, `StandardScaler`, and `RandomForestClassifier`.
- Scalable to support more protocols.

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

## 🖥️ Personalized UI

- Clean **Streamlit interface**
- Intuitive **Sidebar** for switching between modules
- Confidence-based output labels (🟢 High, 🟡 Moderate, 🔴 Low)
- Toggle section for **advanced evaluation** if needed
- ⚡ Light-weight and fast execution

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
- **Streamlit** – Web dashboard UI
- **scikit-learn** – Machine learning models
- **pandas**, **numpy** – Data handling
- **joblib** – Model serialization
- **StandardScaler**, **LabelEncoder** – Preprocessing tools
- **seaborn**, **matplotlib** - Plots in notebooks

---

## 📁 Folder Structure

AI-Network-Security/<br>
│<br>
├── app.py                          # Streamlit frontend <br>
├── README.md                       # Project overview<br>
├── requirements.txt                # Dependencies<br>
│<br>
├── model/                          # Trained ML models<br>
│   ├── threat_model.pkl<br>
│   ├── threat_label_encoder.pkl<br>
│   ├── traffic_classifier.pkl<br>
│   ├── traffic_label_encoder.pkl<br>
│   └── traffic_scaler.pkl<br>
│<br>
├── data/<br>
│   ├── threat_data.csv             # Final balanced URL dataset<br>
│   └── traffic_data.csv            # 87-feature flow dataset<br>
│<br>
├── scripts/<br>
│   ├── train_threat_model.py       # Train threat detection model<br>
│   ├── train_traffic_model.py      # Train traffic classifier<br>
│   ├── evaluate_test_data.py       # Evaluate on test set (optional)<br>
│   └── feature_extraction.py       # Custom feature extractor<br>
│<br>
├── notebooks/<br>
│   ├── evaluate_train_data.ipynb   # Visual analysis of threat model<br>
│   └── traffic_model_training.ipynb # Traffic model training + CM<br>

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

## 👨‍💻 Developed By <br>
### Snith Shibu <br>
### Sidharth Sumitra Gireesh <br>
### Devananda S.R. <br>
ECE students from Mar Baselios College of Engineering and Technology, Trivandrum, Kerala

---

## 📌 Notes
- All models can be retrained via the scripts/ folder.
- traffic_data.csv (the traning data for traffic classification was huge), so wasn't able to commit to the repository
- evaluate_test_data.py can be used later to evaluate new URLs. Save it under the name "sample_test_http.csv"
