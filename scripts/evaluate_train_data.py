import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.feature_extraction import extract_features

# Load model and label encoder
model = joblib.load("model/threat_model.pkl")
le = joblib.load("model/threat_label_encoder.pkl")

# Load dataset
df = pd.read_csv("data/threat_data.csv")  # This is your training data

# Extract features and labels
X = [extract_features(url) for url in df["url"]]
y_true = le.transform(df["label"])   # Encode labels to match y_pred
y_pred = model.predict(X)

# Evaluation
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
