import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("data/traffic_data.csv")

# Filter top 5 traffic classes
top_classes = df['ProtocolName'].value_counts().head(5).index.tolist()
df = df[df['ProtocolName'].isin(top_classes)]

# Drop rows with NaN/inf
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Sample smaller subset for faster training (adjust if needed)
df = df.sample(n=100000, random_state=42)

# Feature selection
X = df.drop(columns=['ProtocolName', 'Flow.ID', 'Source.IP', 'Destination.IP', 'Timestamp'])
y = df['ProtocolName']

# Label encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model artifacts
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/traffic_classifier.pkl")
joblib.dump(scaler, "model/traffic_scaler.pkl")
joblib.dump(le, "model/traffic_label_encoder.pkl")