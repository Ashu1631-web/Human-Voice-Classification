import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("../data/sample_data.csv")   # change if needed

# =========================
# PREPROCESS
# =========================
X = df.drop("label", axis=1)
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN CLASSIFIER
# =========================
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Classifier Accuracy: {acc*100:.2f}%")

# =========================
# TRAIN CLUSTERING
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# =========================
# SAVE MODELS
# =========================
pickle.dump(clf, open("../models/classifier.pkl", "wb"))
pickle.dump(kmeans, open("../models/kmeans.pkl", "wb"))
pickle.dump(scaler, open("../models/scaler.pkl", "wb"))

print("✅ Models saved in models/ folder")
