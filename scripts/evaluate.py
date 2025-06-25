import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.feature_extraction import extract_features

# Load data & model
df = pd.read_csv("sample_http.csv")
model = joblib.load("model/rf_model.pkl")

# Feature extraction
X = [extract_features(url) for url in df["url"]]
y_true = df["label"].values
y_pred = model.predict(X)

# Evaluation
print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
