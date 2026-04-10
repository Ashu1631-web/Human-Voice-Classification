import streamlit as st
import numpy as np
import pandas as pd
import pickle
import librosa
import plotly.express as px
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Voice AI Pro", layout="wide")

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False
    st.session_state.role = None

def login():
    st.title("🔐 Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "admin123":
            st.session_state.login = True
            st.session_state.role = "admin"
        elif user == "user" and pwd == "user123":
            st.session_state.login = True
            st.session_state.role = "user"
        else:
            st.error("Invalid credentials")

def logout():
    st.session_state.login = False

# ---------------- AUDIO FEATURE ----------------
def extract_features(file):
    y, sr = librosa.load(file, sr=None)

    features = [
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.feature.rms(y=y)),
        np.mean(librosa.yin(y, fmin=20, fmax=300))
    ]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    for m in mfcc:
        features.append(np.mean(m))

    return np.array(features).reshape(1, -1)

# ---------------- RECORD AUDIO ----------------
def record_audio():
    fs = 22050
    st.info("Recording...")
    rec = sd.rec(int(5 * fs), samplerate=fs, channels=1)
    sd.wait()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp.name, fs, rec)
    return temp.name

# ---------------- LOAD MODELS ----------------
def load_model(path):
    try:
        return pickle.load(open(path, "rb"))
    except:
        return None

clf = load_model("models/classifier.pkl")
kmeans = load_model("models/kmeans.pkl")

# ---------------- MAIN ----------------
if not st.session_state.login:
    login()

else:
    st.sidebar.title("🎙️ Voice AI Pro")

    menu = st.sidebar.radio("Menu", [
        "Overview",
        "Audio Prediction",
        "EDA Dashboard",
        "Model Dashboard",
        "Retrain Model"
    ])

    if st.sidebar.button("Logout"):
        logout()

    # ---------- OVERVIEW ----------
    if menu == "Overview":
        st.title("📌 Project Overview")

        st.markdown("""
### 🎯 Objective
AI system for voice classification and clustering.

### 🔥 Features
- Login System  
- Audio Upload + Recording  
- ML Prediction  
- Dashboard  
- Model Retraining  

### 💼 Use Cases
- Call center analytics  
- Gender detection  
- Voice AI systems  
""")

    # ---------- AUDIO ----------
    elif menu == "Audio Prediction":

        st.title("🎤 Audio Prediction")

        file = st.file_uploader("Upload WAV", type=["wav"])

        if file:
            st.audio(file)

            if st.button("Analyze"):
                f = extract_features(file)

                pred = clf.predict(f)[0] if clf else 0
                cluster = kmeans.predict(f)[0] if kmeans else 0

                st.success(f"Prediction: {'Male' if pred==1 else 'Female'}")
                st.info(f"Cluster: {cluster}")

        st.subheader("🎙️ Live Recording")

        if st.button("Record Audio"):
            path = record_audio()
            st.audio(path)

            f = extract_features(path)

            pred = clf.predict(f)[0] if clf else 0
            cluster = kmeans.predict(f)[0] if kmeans else 0

            st.success(f"Prediction: {'Male' if pred==1 else 'Female'}")
            st.info(f"Cluster: {cluster}")

    # ---------- EDA ----------
    elif menu == "EDA Dashboard":
        st.title("📊 EDA Dashboard")

        df = pd.read_csv("data/sample_data.csv")

        st.plotly_chart(px.histogram(df, x=df.columns[0]))
        st.plotly_chart(px.scatter(df, x=df.columns[0], y=df.columns[1]))
        st.plotly_chart(px.box(df))
        st.plotly_chart(px.imshow(df.corr()))

    # ---------- MODEL DASHBOARD ----------
    elif menu == "Model Dashboard":

        st.title("📈 Model Performance")

        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)

        acc = accuracy_score(y_true, y_pred)
        st.metric("Accuracy", f"{acc*100:.2f}%")

        cm = confusion_matrix(y_true, y_pred)
        st.plotly_chart(px.imshow(cm, text_auto=True))

    # ---------- RETRAIN ----------
    elif menu == "Retrain Model":

        st.title("📁 Retrain Model")

        file = st.file_uploader("Upload CSV")

        if file:
            df = pd.read_csv(file)
            X = df.drop("label", axis=1)
            y = df["label"]

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)

            st.success(f"Accuracy: {acc*100:.2f}%")

            pickle.dump(model, open("models/classifier.pkl", "wb"))
            st.info("Model saved")
