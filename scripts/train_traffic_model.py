#STEP 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#STEP 2: Load dataset
df = pd.read_csv("data/traffic_data.csv", nrows=100000, quotechar='"')
df = df[df['ProtocolName'].isin(['HTTP', 'SSL', 'GOOGLE', 'HTTP_PROXY', 'HTTP_CONNECT'])]

#STEP 3: Select proper label column
print(df.columns)
label_col = "ProtocolName"

#STEP 4: Preview top traffic types
print(df[label_col].value_counts().head(10))

#STEP 5: Filter for top 5 traffic types and sample 1000 rows each
top_classes = df[label_col].value_counts().head(5).index.tolist()
df_filtered = df[df[label_col].isin(top_classes)]
df_sampled = df_filtered.groupby(label_col).sample(n=1000, random_state=42).reset_index(drop=True)

#STEP 6: Prepare features and labels
labels = df_sampled[label_col]

drop_cols = ['Flow.ID', 'Source.IP', 'Destination.IP', 'Timestamp', 
             'Label', 'L7Protocol', 'ProtocolName']

X = df_sampled.drop(columns=[col for col in drop_cols if col in df_sampled.columns])
X = X.select_dtypes(include=[np.number])  # Keep numeric features only

#STEP 7: Handle missing values
X = X.replace([np.inf, -np.inf], np.nan).dropna()
labels = labels.loc[X.index]  # Ensure labels and features match after cleanup

#STEP 8: Encode target labels
le = LabelEncoder()
y = le.fit_transform(labels)

#STEP 9: Train/test split and feature scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#STEP 10: Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

#STEP 11: Evaluate model
y_pred = clf.predict(X_test_scaled)

# Show classification report
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, cmap="Blues", 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#STEP 12: Save model components
joblib.dump(clf, "model/traffic_classifier.pkl")
joblib.dump(le, "model/traffic_label_encoder.pkl")
joblib.dump(scaler, "model/traffic_scaler.pkl")
