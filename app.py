import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Clustering Classification", layout="wide")

# ================= UI =================
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1610733661495-4aa6ed9fc6f4?q=80&w=869&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
}
.glass {
    background: rgba(0,0,0,0.7);
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🔐 Human Clustering Classification Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if (u=="admin" and p=="admin123") or (u=="user" and p=="user123"):
            st.session_state.login = True
        else:
            st.error("Invalid credentials ❌")

# ================= MODEL =================
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    return model, scaler

model, scaler = load_models()

# ================= DATA =================
@st.cache_data
def load_data():
    if os.path.exists("vocal_gender_features_new.csv"):
        return pd.read_csv("vocal_gender_features_new.csv")
    return None

# ================= FEATURES =================
def extract_features(file):
    import librosa
    y, sr = librosa.load(file, duration=3)

    f = []
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    f.extend([np.mean(sc), np.std(sc)])

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    f.extend([np.mean(sb), np.std(sb)])

    f.append(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    f.append(np.mean(librosa.feature.spectral_flatness(y=y)))
    f.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    f.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    f.append(np.mean(librosa.feature.rms(y=y)))

    pitch = librosa.yin(y, fmin=50, fmax=300)
    f.extend([np.mean(pitch), np.min(pitch), np.max(pitch), np.std(pitch)])

    f.append(skew(y))
    f.append(kurtosis(y))
    f.append(-np.sum(y**2 * np.log(y**2 + 1e-10)))
    f.append(np.log(np.sum(y**2) + 1e-10))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        f.append(np.mean(mfcc[i]))
        f.append(np.std(mfcc[i]))

    return np.array(f).reshape(1, -1)

FEATURE_COLUMNS = [
 'mean_spectral_centroid','std_spectral_centroid','mean_spectral_bandwidth','std_spectral_bandwidth',
 'mean_spectral_contrast','mean_spectral_flatness','mean_spectral_rolloff','zero_crossing_rate',
 'rms_energy','mean_pitch','min_pitch','max_pitch','std_pitch',
 'spectral_skew','spectral_kurtosis','energy_entropy','log_energy',
 'mfcc_1_mean','mfcc_1_std','mfcc_2_mean','mfcc_2_std','mfcc_3_mean','mfcc_3_std',
 'mfcc_4_mean','mfcc_4_std','mfcc_5_mean','mfcc_5_std','mfcc_6_mean','mfcc_6_std',
 'mfcc_7_mean','mfcc_7_std','mfcc_8_mean','mfcc_8_std','mfcc_9_mean','mfcc_9_std',
 'mfcc_10_mean','mfcc_10_std','mfcc_11_mean','mfcc_11_std','mfcc_12_mean','mfcc_12_std',
 'mfcc_13_mean','mfcc_13_std'
]

# ================= MAIN =================
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("Menu", ["Overview","Audio","EDA","Model","Clustering"])

# ================= OVERVIEW (UPDATED ONLY) =================
if menu=="Overview":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.title("👥 Human Voice Clustering AI")

    st.markdown("""
### 📌 Project Overview

This is an AI-powered system that analyzes human voice audio and performs:

- 🎤 Gender Detection (Male / Female)
- 👥 Human Voice Clustering
- 📊 Data Analysis & Visualization

---

### 🚀 Features

✔ Upload audio  
✔ Live recording  
✔ Gender prediction  
✔ EDA dashboard (10 graphs)  
✔ Model insights  
✔ Clustering system  

---

### 🔄 Workflow

Audio → Feature Extraction → Scaling → Model → Prediction  

---

### 🛠 Tech Stack

Python | Streamlit | Librosa | Scikit-learn | Plotly  

---

### 💼 Use Cases

Call center analytics, AI voice systems, clustering & classification  
""")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= AUDIO =================
elif menu=="Audio":
    file = st.file_uploader("Upload Audio (.wav / mp3)", type=["wav","mp3"])

    if file:
        with open("temp.wav","wb") as f:
            f.write(file.getbuffer())

        st.audio("temp.wav")

        if st.button("Analyze"):
            f = extract_features("temp.wav")
            df = pd.DataFrame(f, columns=FEATURE_COLUMNS)
            f = scaler.transform(df)

            proba = model.predict_proba(f)
            female_prob = proba[0][0]
            male_prob = proba[0][1]

            if female_prob > 0.6:
                result = "Female 👩"
            elif male_prob > 0.6:
                result = "Male 👨"
            else:
                result = "Uncertain ⚠️"

            st.success(f"🎯 Prediction: {result}")
            st.info(f"Confidence: {proba}")

    st.markdown("### 🎤 Record Voice")
    audio = st.audio_input("Click to record")

    if audio:
        with open("live.wav","wb") as f:
            f.write(audio.getbuffer())

        st.audio("live.wav")

        f = extract_features("live.wav")
        df = pd.DataFrame(f, columns=FEATURE_COLUMNS)
        f = scaler.transform(df)

        proba = model.predict_proba(f)

        female_prob = proba[0][0]
        male_prob = proba[0][1]

        if female_prob > 0.6:
            result = "Female 👩"
        elif male_prob > 0.6:
            result = "Male 👨"
        else:
            result = "Uncertain ⚠️"

        st.success(f"🎯 Prediction: {result}")
        st.info(f"Confidence: {proba}")

# ================= EDA =================
elif menu=="EDA":
    df = load_data()

    if df is not None:
        st.title("📊 Data Analysis")

        st.plotly_chart(px.histogram(df, x="label"))
        st.plotly_chart(px.histogram(df, x="mean_pitch"))
        st.plotly_chart(px.histogram(df, x="mean_spectral_centroid"))
        st.plotly_chart(px.histogram(df, x="mean_spectral_bandwidth"))
        st.plotly_chart(px.histogram(df, x="rms_energy"))
        st.plotly_chart(px.box(df, x="label", y="mean_pitch"))
        st.plotly_chart(px.histogram(df, x="mfcc_1_mean"))
        st.plotly_chart(px.histogram(df, x="mfcc_2_mean"))
        st.plotly_chart(px.imshow(df.corr()))
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="label"))

# ================= MODEL =================
elif menu=="Model":
    df = load_data()

    if df is not None:
        st.bar_chart(df.drop("label", axis=1).mean().sort_values(ascending=False).head(10))
        st.plotly_chart(px.histogram(df, x="label"))
        st.plotly_chart(px.histogram(df, x="mean_pitch"))
        st.plotly_chart(px.histogram(df, x="rms_energy"))
        st.plotly_chart(px.box(df, x="label", y="mean_pitch"))
        st.plotly_chart(px.imshow(df.corr()))

# ================= CLUSTERING =================
elif menu=="Clustering":
    df = load_data()

    if df is not None:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = df.drop("label", axis=1)
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=2, random_state=42)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        st.plotly_chart(px.histogram(df, x="cluster"))
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="cluster"))
        st.plotly_chart(px.histogram(df, x="cluster", color="label"))
