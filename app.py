import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils.feature_extraction import extract_features

# Load models
threat_model = joblib.load("model/rf_model.pkl")
traffic_model = joblib.load("../model/traffic_classifier.pkl")
traffic_scaler = joblib.load("../model/traffic_scaler.pkl")
traffic_label_encoder = joblib.load("../model/traffic_label_encoder.pkl")
threat_label_encoder = joblib.load("../model/rf_label_encoder.pkl")

# App layout
st.set_page_config(page_title="AI Network Security", layout="wide")
st.title("🛡️ AI-Powered Network Security Tool")

tabs = st.tabs(["🚨 Threat Detection", "🚦 Traffic Classification"])

# ==================== THREAT DETECTION TAB ==================== #
with tabs[0]:
    st.header("Detect SQL Injection / XSS Attacks")

    url_input = st.text_input("🔗 Enter a single URL:", placeholder="http://example.com?id=1 OR 1=1")

    if st.button("Analyze URL"):
        features = extract_features(url_input).reshape(1, -1)
        pred = threat_model.predict(features)[0]
        label = threat_label_encoder.inverse_transform([pred])[0]
        
        if label == "benign":
            st.success(f"✅ Benign URL")
        else:
            st.error(f"⚠️ Malicious — {label.upper()}")

    st.divider()
    st.subheader("📁 Upload CSV with 'url' and 'label'")

    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if 'url' in df.columns:
            df['predicted'] = df['url'].apply(lambda u: threat_label_encoder.inverse_transform(
                threat_model.predict([extract_features(u)])[0:1])[0])
            st.dataframe(df[['url', 'predicted']])
            csv_out = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results", csv_out, "threat_predictions.csv", "text/csv")
        else:
            st.warning("CSV must contain a 'url' column.")

# ==================== TRAFFIC CLASSIFICATION TAB ==================== #
with tabs[1]:
    st.header("Classify Network Traffic Flows")

    st.subheader("📁 Upload flow-level CSV")
    file2 = st.file_uploader("Upload network flow data", type=['csv'], key="traffic")

    if file2:
        df2 = pd.read_csv(file2)

        # Clean and prepare
        ignore_cols = ['Flow.ID', 'Source.IP', 'Destination.IP', 'Timestamp',
                       'Label', 'L7Protocol', 'ProtocolName']

        X = df2.drop(columns=[col for col in ignore_cols if col in df2.columns], errors='ignore')
        X = X.select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan).dropna()

        # Ensure enough rows exist after cleaning
        if X.empty:
            st.error("All rows were dropped after cleaning. Check your CSV.")
        else:
            X_scaled = traffic_scaler.transform(X)
            preds = traffic_model.predict(X_scaled)
            class_names = traffic_label_encoder.inverse_transform(preds)

            df2 = df2.loc[X.index]  # align indexes
            df2['Predicted_Class'] = class_names
            st.dataframe(df2[['Predicted_Class'] + [col for col in df2.columns if col not in ignore_cols][:5]])

            csv_out = df2.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Classification Results", csv_out, "traffic_predictions.csv", "text/csv")

