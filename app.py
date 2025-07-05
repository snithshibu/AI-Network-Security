import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scripts.feature_extraction import extract_features
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load models
threat_model = joblib.load("model/threat_model.pkl")
threat_label_encoder = joblib.load("model/threat_label_encoder.pkl")
traffic_model = joblib.load("model/traffic_classifier.pkl")
traffic_label_encoder = joblib.load("model/traffic_label_encoder.pkl")
traffic_scaler = joblib.load("model/traffic_scaler.pkl")

# Page config
st.set_page_config(page_title="AI Network Security", layout="wide", page_icon="🛡️")

# -------------------------------
# Sidebar 
# -------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://img.icons8.com/fluency/96/shield.png" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("## 🛡️ AI Network Security")
    st.markdown(
        """
        **Overview**
        
        This tool uses AI/ML models to:
        - Detect threats like XSS and SQL Injection in URLs
        - Classify network traffic flows by application type
        - Perform real-time & batch analysis of URLs and traffic data
        - Show confidence levels for predictions
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        **Built by:**  
        Snith Shibu  
        Sidharth Sumitra Gireesh  
        Devananda S. R.
        """,
        unsafe_allow_html=True
    )
    st.markdown("📚 Intel® Unnati Training Program 2025")
    st.markdown("---")
    st.markdown("🔗 [GitHub Repo](https://github.com/snithshibu/AI-Network-Security)")

# -------------------------------
# Main Tabs
# -------------------------------
st.title("🧠 AI-Powered Network Security Dashboard")
tabs = st.tabs(["🔍 Threat Detection (XSS / SQLi)", "🚦 Traffic Flow Classification"])

# -------------------------------
# 🔍 Tab 1: Threat Detection
# -------------------------------
with tabs[0]:
    st.header("🔗 URL Analysis for Threats")

    url_input = st.text_input("Enter a URL:", placeholder="http://example.com?id=1 OR 1=1")

    if st.button("Analyze URL"):
        if url_input.strip() == "":
            st.warning("Please enter a valid URL.")
        else:
            try:
                features = extract_features(url_input).reshape(1, -1)
                probs = threat_model.predict_proba(features)[0]
                pred_idx = probs.argmax()
                label = str(threat_label_encoder.inverse_transform([pred_idx])[0])
                confidence = round(probs[pred_idx] * 100, 2)

                # Confidence coloring
                if confidence >= 90:
                    badge_color = "🟢 High"
                elif confidence >= 70:
                    badge_color = "🟡 Moderate"
                else:
                    badge_color = "🔴 Low"
                    st.warning("⚠️ Confidence is low — the model may be unsure. Consider double-checking manually.")

                st.markdown(f"**🧠 Confidence:** {confidence}% ({badge_color})")

                if label.lower() == "benign":
                    st.success(f"✅ Benign URL")
                else:
                    st.error(f"⚠️ Malicious — {label.upper()}")
            except Exception as e:
                st.exception(e)

    st.divider()

    st.subheader("📁 Batch URL Prediction via CSV")
    st.markdown("Upload a CSV with a `url` column.")

    file = st.file_uploader("Choose a CSV", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file)
            if 'url' not in df.columns:
                st.error("❌ Your CSV must contain a 'url' column.")
            else:
                df['predicted'] = df['url'].apply(
                    lambda u: str(threat_label_encoder.inverse_transform(
                        threat_model.predict([extract_features(u)])[0:1])[0])
                )

                df['confidence'] = df['url'].apply(
                    lambda u: round(max(threat_model.predict_proba([extract_features(u)])[0]) * 100, 2)
                )

                st.markdown("### 🧾 Results")
                st.dataframe(df[['url', 'predicted', 'confidence']])

                avg_conf = df['confidence'].mean()
                st.info(f"📈 Average model confidence: **{round(avg_conf, 2)}%**")

                csv_out = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results", csv_out, "threat_predictions.csv", "text/csv")
        except Exception as e:
            st.exception(e)

# -------------------------------
# 🚦 Tab 2: Traffic Classification
# -------------------------------
with tabs[1]:
    st.header("📶 Application Traffic Flow Classifier")

    st.markdown("Upload a CSV containing **87 network flow metadata features**.")

    traffic_file = st.file_uploader("Upload traffic metadata CSV", type=["csv"], key="traffic")
    if traffic_file:
        try:
            traffic_df = pd.read_csv(traffic_file)

            # Extract numeric-only features for prediction
            traffic_features = traffic_df.select_dtypes(include=np.number)
            X_scaled = traffic_scaler.transform(traffic_features)
            preds = traffic_model.predict(X_scaled)
            labels = traffic_label_encoder.inverse_transform(preds)

            traffic_df['Predicted_App'] = labels

            st.markdown("### 🔍 Classification Output")
            st.dataframe(traffic_df[['Predicted_App']].join(traffic_df.drop('Predicted_App', axis=1, errors='ignore')))

            csv_out = traffic_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Classified Traffic", csv_out, "classified_traffic.csv", "text/csv")

        except Exception as e:
            st.exception(e)
