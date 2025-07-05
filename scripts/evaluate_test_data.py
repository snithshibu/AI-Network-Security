import pandas as pd
import joblib
from scripts.feature_extraction import extract_features
from sklearn.metrics import classification_report

# Load model + label encoder
threat_model = joblib.load("model/threat_model.pkl")
threat_label_encoder = joblib.load("model/threat_label_encoder.pkl")

# Load test data (ensure this path is correct)
df = pd.read_csv("data/sample_threat_data.csv")  
X = [extract_features(url) for url in df["url"]]
y_true = df["label"]

# Encode true labels
y_encoded = threat_label_encoder.transform(y_true)

# Predict
y_pred = threat_model.predict(X)

# Print metrics
print("\n🎯 Classification Report:")
print(classification_report(y_encoded, y_pred, target_names=threat_label_encoder.classes_))
