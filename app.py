import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import librosa
import os

st.set_page_config(page_title="Human Voice AI", layout="wide")

# ===== LOGIN BG ONLY =====
def login():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1511376777868-611b54f68947");
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u=="admin" and p=="admin123":
            st.session_state.login=True
        else:
            st.error("Invalid credentials")

if "login" not in st.session_state:
    st.session_state.login=False

# ===== LOAD MODEL =====
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ===== DATA =====
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

# ===== FEATURE EXTRACTION =====
def extract_features(file):
    y, sr = librosa.load(file, duration=3)

    features = []

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([np.mean(sc), np.std(sc)])

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.extend([np.mean(sb), np.std(sb)])

    features.append(np.mean(librosa.feature.rms(y=y)))

    pitch = librosa.yin(y, fmin=80, fmax=250)
    pitch = pitch[~np.isnan(pitch)]
    if len(pitch)==0:
        pitch=[0]

    features.extend([np.mean(pitch), np.std(pitch)])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    for i in range(5):
        features.append(np.mean(mfcc[i]))

    return np.array(features).reshape(1,-1)

FEATURE_COLUMNS = load_data().drop("label",axis=1).columns

# ===== MAIN =====
if not st.session_state.login:
    login()
    st.stop()

# remove bg after login
st.markdown("<style>.stApp {background:none;}</style>", unsafe_allow_html=True)

menu = st.sidebar.radio("Menu", ["Overview","Audio","Recording","EDA","Classification"])

# ===== OVERVIEW =====
if menu=="Overview":
    st.title("👥 Human Voice Classification System")

    st.markdown("""
### 📌 Project Overview

AI system for:

- 🎤 Gender Detection (Male/Female)
- 📊 EDA Visualization
- 🤖 Classification Dashboard

---

### 🚀 Features

✔ Upload audio  
✔ Live recording  
✔ Gender prediction  
✔ 10 EDA graphs  
✔ CSV table + filter  

---

### 🔄 Workflow

Audio → Feature Extraction → Scaling → Model → Prediction  

---

### 🛠 Tech Stack

Python | Streamlit | Librosa | Scikit-learn | Plotly
""")

# ===== AUDIO =====
elif menu=="Audio":
    file = st.file_uploader("Upload Audio", type=["wav","mp3"])

    if file:
        with open("temp.wav","wb") as f:
            f.write(file.getbuffer())

        st.audio("temp.wav")

        if st.button("Analyze"):
            f = extract_features("temp.wav")
            df = pd.DataFrame(f, columns=FEATURE_COLUMNS)
            f = scaler.transform(df)

            proba = model.predict_proba(f)

            female = proba[0][0]
            male = proba[0][1]

            if female > 0.6:
                res = "Female 👩"
            elif male > 0.6:
                res = "Male 👨"
            else:
                res = "Uncertain ⚠️"

            st.success(res)
            st.write(proba)

# ===== RECORDING =====
elif menu=="Recording":
    audio = st.audio_input("Record")

    if audio:
        with open("live.wav","wb") as f:
            f.write(audio.getbuffer())

        st.audio("live.wav")

        f = extract_features("live.wav")
        df = pd.DataFrame(f, columns=FEATURE_COLUMNS)
        f = scaler.transform(df)

        proba = model.predict_proba(f)

        female = proba[0][0]
        male = proba[0][1]

        if female > 0.6:
            res = "Female 👩"
        elif male > 0.6:
            res = "Male 👨"
        else:
            res = "Uncertain ⚠️"

        st.success(res)
        st.write(proba)

# ===== EDA =====
elif menu=="EDA":
    df = load_data()
    color_map={"male":"red","female":"pink"}

    st.plotly_chart(px.histogram(df,x="label",color="label",title="1. Gender Distribution",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_pitch",color="label",title="2. Pitch Distribution",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="rms_energy",color="label",title="3. Energy Distribution",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_spectral_centroid",color="label",title="4. Spectral Centroid",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_spectral_bandwidth",color="label",title="5. Spectral Bandwidth",color_discrete_map=color_map))
    st.plotly_chart(px.box(df,x="label",y="mean_pitch",color="label",title="6. Pitch Boxplot",color_discrete_map=color_map))
    st.plotly_chart(px.box(df,x="label",y="rms_energy",color="label",title="7. Energy Boxplot",color_discrete_map=color_map))
    st.plotly_chart(px.scatter(df,x="mean_pitch",y="rms_energy",color="label",title="8. Pitch vs Energy",color_discrete_map=color_map))
    st.plotly_chart(px.imshow(df.corr(),title="9. Correlation Heatmap"))
    st.plotly_chart(px.histogram(df,x="mfcc_1_mean",color="label",title="10. MFCC1",color_discrete_map=color_map))

# ===== CLASSIFICATION =====
elif menu=="Classification":
    df = load_data()

    filt = st.selectbox("Filter",["All","male","female"])
    if filt!="All":
        df = df[df["label"]==filt]

    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False))

    color_map={"male":"red","female":"pink"}

    st.plotly_chart(px.histogram(df,x="label",color="label",title="1. Class Distribution",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_pitch",color="label",title="2. Pitch",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="rms_energy",color="label",title="3. Energy",color_discrete_map=color_map))
    st.plotly_chart(px.scatter(df,x="mean_pitch",y="rms_energy",color="label",title="4. Scatter",color_discrete_map=color_map))
    st.plotly_chart(px.box(df,x="label",y="mean_pitch",color="label",title="5. Pitch Box",color_discrete_map=color_map))
    st.plotly_chart(px.box(df,x="label",y="rms_energy",color="label",title="6. Energy Box",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mfcc_1_mean",color="label",title="7. MFCC1",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mfcc_2_mean",color="label",title="8. MFCC2",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mfcc_3_mean",color="label",title="9. MFCC3",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mfcc_4_mean",color="label",title="10. MFCC4",color_discrete_map=color_map))
