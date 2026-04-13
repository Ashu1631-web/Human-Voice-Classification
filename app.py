import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Clustering Classification", layout="wide")

# ================= SIDEBAR ANIMATION =================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
}
[data-testid="stSidebar"]::after {
    content: "";
    position: absolute;
    width: 100%;
    height: 100%;
    background: repeating-linear-gradient(
        90deg,
        rgba(255,255,255,0.05) 0px,
        rgba(255,255,255,0.05) 2px,
        transparent 2px,
        transparent 6px
    );
    animation: move 2s linear infinite;
}
@keyframes move {
    0% {transform: translateY(0);}
    100% {transform: translateY(20px);}
}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1610733661495-4aa6ed9fc6f4?q=80&w=869&auto=format&fit=crop");
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

def login():
    st.title("🔐 Human Clustering Classification Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if (u=="admin" and p=="admin123") or (u=="user" and p=="user123"):
            st.session_state.login = True
        else:
            st.error("Invalid credentials ❌")

# ================= LOAD =================
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    return model, scaler

model, scaler = load_models()

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

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        f.append(np.mean(mfcc[i]))
        f.append(np.std(mfcc[i]))

    return np.array(f).reshape(1, -1)

# ================= MAIN =================
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("Menu", ["Overview","EDA","Classification","Clustering","Audio"])

# ================= OVERVIEW (UNCHANGED) =================
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

# ================= EDA =================
elif menu=="EDA":
    df = load_data()
    if df is not None:
        df["label"] = df["label"].map({0:"female",1:"male"})

        st.title("📊 Exploratory Data Analysis")
        st.dataframe(df.head(20))

        for i, col in enumerate(df.drop("label", axis=1).columns[:10]):
            st.plotly_chart(px.histogram(
                df, x=col, color="label",
                title=f"{i+1}. {col}",
                template="plotly_dark"
            ))

# ================= CLASSIFICATION =================
elif menu=="Classification":
    st.title("🧠 Classification")

    df = load_data()
    if df is not None:
        df["label"] = df["label"].map({0:"female",1:"male"})
        st.dataframe(df.head(20))

        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        X = df.drop("label", axis=1)
        y = df["label"]

        X_scaled = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

        model_svm = SVC(kernel="rbf", C=5, probability=True)
        model_svm.fit(X_train, y_train)

        st.success(f"Accuracy: {round(model_svm.score(X_test,y_test)*100,2)}%")

        for i, col in enumerate(X.columns[:10]):
            st.plotly_chart(px.histogram(
                df, x=col, color="label",
                title=f"{i+1}. {col}",
                template="plotly_dark"
            ))

# ================= CLUSTERING =================
elif menu=="Clustering":
    st.title("🔍 Clustering")

    df = load_data()
    if df is not None:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        X = df.drop("label", axis=1)
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(X_scaled)

        st.write("Score:", silhouette_score(X_scaled, labels))

        df["cluster"] = labels

        st.plotly_chart(px.histogram(df, x="cluster"))
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="cluster"))
        st.plotly_chart(px.histogram(df, x="cluster", color="label"))
        st.plotly_chart(px.box(df, x="cluster", y="mean_pitch"))
        st.plotly_chart(px.box(df, x="cluster", y="rms_energy"))
        st.plotly_chart(px.scatter(df, x="mean_spectral_centroid", y="mean_spectral_bandwidth", color="cluster"))

# ================= AUDIO =================
elif menu=="Audio":
    file = st.file_uploader("Upload Audio", type=["wav","mp3"])

    if file:
        with open("temp.wav","wb") as f:
            f.write(file.getbuffer())

        st.audio("temp.wav")

        f = extract_features("temp.wav")
        df_feat = pd.DataFrame(f, columns=load_data().drop("label", axis=1).columns)
        f_scaled = scaler.transform(df_feat)

        proba = model.predict_proba(f_scaled)
        classes = model.classes_

        result = dict(zip(classes, proba[0]))

        st.success(f"🎯 Prediction: {max(result, key=result.get)}")
        st.info(result)

    st.markdown("### 🎤 Record Voice")
    audio = st.audio_input("Record")

    if audio:
        with open("live.wav","wb") as f:
            f.write(audio.getbuffer())

        st.audio("live.wav")
