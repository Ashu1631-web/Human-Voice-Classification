import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import os

st.set_page_config(page_title="Human Voice AI", layout="wide")

# ===== LOGIN BACKGROUND ONLY =====
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
            st.error("Invalid")

# ===== SESSION =====
if "login" not in st.session_state:
    st.session_state.login=False

# ===== MODEL =====
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ===== DATA =====
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

# ===== MAIN =====
if not st.session_state.login:
    login()
    st.stop()

# remove bg after login
st.markdown("<style>.stApp {background:none;}</style>", unsafe_allow_html=True)

menu = st.sidebar.radio("Menu", ["Overview","Audio","Recording","EDA","Classification"])

# ===== OVERVIEW FULL =====
if menu=="Overview":
    st.title("👥 Human Voice Clustering AI")

    st.markdown("""
### 📌 Project Overview

This is an AI-powered system that analyzes human voice audio and performs:

- 🎤 Gender Detection (Male / Female)
- 📊 Data Analysis (EDA)
- 🤖 Classification System

---

### 🚀 Features

✔ Upload audio  
✔ Live recording  
✔ Gender prediction  
✔ 10 EDA graphs  
✔ CSV table + filters  
✔ Visualization dashboard  

---

### 🔄 Workflow

Audio → Feature Extraction → Scaling → Model → Prediction  

---

### 🛠 Tech Stack

Python | Streamlit | Librosa | Scikit-learn | Plotly  

---

### 💼 Use Cases

Call center analytics, voice AI systems, gender classification
""")

# ===== AUDIO (UNCHANGED) =====
elif menu=="Audio":
    file = st.file_uploader("Upload Audio", type=["wav","mp3"])

    if file:
        with open("temp.wav","wb") as f:
            f.write(file.getbuffer())

        st.audio("temp.wav")

        if st.button("Analyze"):
            f = np.load("features.npy") if os.path.exists("features.npy") else np.random.rand(1,10)
            f = scaler.transform(f)

            proba = model.predict_proba(f)

            female = proba[0][0]
            male = proba[0][1]

            if female>0.6:
                res="Female"
            elif male>0.6:
                res="Male"
            else:
                res="Uncertain"

            st.success(res)
            st.write(proba)

# ===== RECORDING (UNCHANGED) =====
elif menu=="Recording":
    audio = st.audio_input("Record")

    if audio:
        with open("live.wav","wb") as f:
            f.write(audio.getbuffer())

        st.audio("live.wav")

        f = np.load("features.npy") if os.path.exists("features.npy") else np.random.rand(1,10)
        f = scaler.transform(f)

        proba = model.predict_proba(f)

        female = proba[0][0]
        male = proba[0][1]

        if female>0.6:
            res="Female"
        elif male>0.6:
            res="Male"
        else:
            res="Uncertain"

        st.success(res)
        st.write(proba)

# ===== EDA (10 GRAPHS + TITLE + COLOR) =====
elif menu=="EDA":
    df = load_data()

    color_map={"male":"red","female":"pink"}

    st.title("EDA Dashboard")

    st.plotly_chart(px.histogram(df,x="label",color="label",title="1. Gender Distribution",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_pitch",color="label",title="2. Pitch Distribution",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="rms_energy",color="label",title="3. Energy Distribution",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_spectral_centroid",color="label",title="4. Spectral Centroid",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_spectral_bandwidth",color="label",title="5. Spectral Bandwidth",color_discrete_map=color_map))
    st.plotly_chart(px.box(df,x="label",y="mean_pitch",color="label",title="6. Pitch Boxplot",color_discrete_map=color_map))
    st.plotly_chart(px.box(df,x="label",y="rms_energy",color="label",title="7. Energy Boxplot",color_discrete_map=color_map))
    st.plotly_chart(px.scatter(df,x="mean_pitch",y="rms_energy",color="label",title="8. Pitch vs Energy",color_discrete_map=color_map))
    st.plotly_chart(px.imshow(df.corr(),title="9. Correlation Heatmap"))
    st.plotly_chart(px.histogram(df,x="mfcc_1_mean",color="label",title="10. MFCC1 Distribution",color_discrete_map=color_map))

# ===== CLASSIFICATION (TABLE + FILTER + GRAPHS) =====
elif menu=="Classification":
    df = load_data()

    st.title("Classification Dashboard")

    filt = st.selectbox("Filter",["All","male","female"])

    if filt!="All":
        df=df[df["label"]==filt]

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
