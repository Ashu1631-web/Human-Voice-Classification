import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis

# ================= CONFIG =================
st.set_page_config(page_title="Voice AI Pro", layout="wide")

# ================= UI =================
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1511376777868-611b54f68947");
    background-size: cover;
}
.glass {
    background: rgba(0,0,0,0.7);
    padding: 20px;
    border-radius: 15px;
}
h1,h2,h3 {color:#00eaff;text-align:center;}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🎙️ Voice AI Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if (u=="admin" and p=="admin123") or (u=="user" and p=="user123"):
            st.session_state.login = True
        else:
            st.error("Invalid credentials")

# ================= DATA =================
@st.cache_data
def load_data():
    import os
    paths = ["data/vocal_gender_features_new.csv","vocal_gender_features_new.csv"]

    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "label" not in df.columns and "gender" in df.columns:
                df.rename(columns={"gender":"label"}, inplace=True)
            return df.select_dtypes(include=np.number)

    st.warning("Upload dataset")
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f:
        return pd.read_csv(f)

    st.stop()

# ================= FIXED AUDIO FEATURES =================
def extract_features(file):
    import librosa

    y, sr = librosa.load(file, duration=3)

    features = []

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([np.mean(sc), np.std(sc)])

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.extend([np.mean(sb), np.std(sb)])

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.append(np.mean(contrast))

    flatness = librosa.feature.spectral_flatness(y=y)
    features.append(np.mean(flatness))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))

    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))

    pitch = librosa.yin(y, fmin=50, fmax=300)
    features.extend([np.mean(pitch), np.min(pitch), np.max(pitch), np.std(pitch)])

    features.append(skew(y))
    features.append(kurtosis(y))
    features.append(-np.sum(y**2 * np.log(y**2 + 1e-10)))
    features.append(np.log(np.sum(y**2) + 1e-10))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features.append(np.mean(mfcc[i]))
        features.append(np.std(mfcc[i]))

    return np.array(features).reshape(1, -1)

# ================= MODEL LOAD =================
def load_model(path):
    try:
        return pickle.load(open(path,"rb"))
    except:
        return None

clf = load_model("model.pkl")
scaler = load_model("scaler.pkl")

# ================= FEATURE COLUMNS =================
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

else:
    menu = st.sidebar.radio("Menu", ["Overview","Audio","EDA","Model"])

    # ---------- AUDIO ----------
    if menu=="Audio":
        file = st.file_uploader("Upload Audio (.wav recommended)", type=["wav","mp3"])

        if file:
            st.audio(file)

            if st.button("Analyze"):
                f = extract_features(file)

                df = pd.DataFrame(f, columns=FEATURE_COLUMNS)

                if scaler:
                    f = scaler.transform(df)

                try:
                    proba = clf.predict_proba(f)
                    pred = clf.predict(f)[0]

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

                except Exception as e:
                    st.error(f"Prediction error: {e}")
