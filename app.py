import streamlit as st
import pandas as pd
import numpy as np
import librosa
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

st.set_page_config(page_title="Voice AI System", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background: #1f2937;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file):
    y, sr = librosa.load(file, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([mfcc, pitch, zcr]).reshape(1, -1)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/features.csv")

df = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train():
    X = df.drop("label", axis=1)
    y = df["label"]

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    km = KMeans(n_clusters=3)
    km.fit(X)

    return clf, km

model, kmeans = train()

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;color:#00C9A7'>
🎙️ Human Voice Classification & Clustering
</h1>
<p style='text-align:center'>Advanced Audio Analytics Dashboard</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

audio_file = st.sidebar.file_uploader("Upload Voice", type=["wav"])

# ---------------- KPI CARDS ----------------
col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='card'>Total Samples<br><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'>Features<br><h2>{df.shape[1]-1}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'>Classes<br><h2>{df['label'].nunique()}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Classification", "🧩 Clustering", "📊 Analytics", "📈 Insights"]
)

# ================= CLASSIFICATION =================
with tab1:
    st.subheader("Voice Classification")

    if audio_file:
        st.audio(audio_file)

        features = extract_features(audio_file)
        pred = model.predict(features)[0]

        st.success(f"Predicted Class: {pred}")

# ================= CLUSTERING =================
with tab2:
    st.subheader("Voice Clustering")

    if audio_file:
        features = extract_features(audio_file)
        cluster = kmeans.predict(features)[0]

        st.info(f"Cluster Group: {cluster}")

# ================= ANALYTICS =================
with tab3:
    st.subheader("Dataset Visualization")

    fig = px.scatter(
        df,
        x=df.columns[0],
        y=df.columns[1],
        color="label",
        title="Voice Clusters"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(df, x="label", title="Class Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# ================= INSIGHTS =================
with tab4:
    st.subheader("Feature Importance")

    importance = df.drop("label", axis=1).mean().sort_values(ascending=False).head(10)

    fig3 = px.bar(
        importance,
        title="Top Features",
        labels={"value": "Importance", "index": "Feature"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Raw Data")
    st.dataframe(df.head(100))
