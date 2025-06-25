import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from utils.feature_extraction import extract_features

# Load the labeled dataset (multi-class)
df = pd.read_csv("sample_http.csv")  # Must have 'url' and 'label' columns: 'benign', 'sql_injection', 'xss'

# Extract features
X = np.array([extract_features(u) for u in df['url']])
y = df['label']

# Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
target_names = [str(cls) for cls in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

# Save model and label encoder
joblib.dump(clf, "model/rf_model.pkl")
joblib.dump(le, "model/rf_label_encoder.pkl")
