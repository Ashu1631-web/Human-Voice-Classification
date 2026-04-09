import joblib

cluster = joblib.load("models/cluster.pkl")

def cluster_voice(features):
    return cluster.predict(features)[0]
