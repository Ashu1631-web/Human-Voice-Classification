import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import joblib

df = pd.read_csv("data/features.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Train classifier
model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "models/classifier.pkl")

# Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

joblib.dump(kmeans, "models/cluster.pkl")
