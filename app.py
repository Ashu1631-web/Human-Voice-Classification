import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="Human Voice AI", layout="wide")

# ===== BLUE GRADIENT SIDEBAR =====
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
</style>
""", unsafe_allow_html=True)

# ===== LOGIN =====
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u=="admin" and p=="admin123":
            st.session_state.login=True

# ===== MODEL =====
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ===== DATA =====
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

# ===== FEATURES =====
def extract_features(file):
    import librosa
    y, sr = librosa.load(file, duration=3)

    f = []
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    f.extend([np.mean(sc), np.std(sc)])

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    f.extend([np.mean(sb), np.std(sb)])

    f.append(np.mean(librosa.feature.rms(y=y)))

    pitch = librosa.yin(y, fmin=80, fmax=250)
    pitch = pitch[~np.isnan(pitch)]
    if len(pitch)==0:
        pitch=[0]

    f.extend([np.mean(pitch), np.std(pitch)])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    for i in range(5):
        f.append(np.mean(mfcc[i]))

    return np.array(f).reshape(1,-1)

FEATURE_COLUMNS = load_data().drop("label",axis=1).columns

# ===== MAIN =====
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("Menu", ["Overview","Audio","Recording","EDA","Classification"])

# ===== OVERVIEW =====
if menu=="Overview":
    st.title("Human Voice Classification System")

# ===== AUDIO (UNCHANGED LOGIC) =====
elif menu=="Audio":
    file = st.file_uploader("Upload", type=["wav","mp3"])
    if file:
        open("temp.wav","wb").write(file.getbuffer())
        st.audio("temp.wav")

        if st.button("Analyze"):
            f = extract_features("temp.wav")
            df = pd.DataFrame(f, columns=FEATURE_COLUMNS)
            f = scaler.transform(df)

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

# ===== RECORDING (FIX ONLY HERE) =====
elif menu=="Recording":
    audio = st.audio_input("Record")

    if audio:
        open("live.wav","wb").write(audio.getbuffer())
        st.audio("live.wav")

        f = extract_features("live.wav")
        df = pd.DataFrame(f, columns=FEATURE_COLUMNS)
        f = scaler.transform(df)

        proba = model.predict_proba(f)

        female = proba[0][0]
        male = proba[0][1]

        # 👉 FIXED LOGIC
        if abs(female-male) < 0.15:
            res="Uncertain"
        else:
            res="Female" if female>male else "Male"

        st.success(res)
        st.write(f"Male: {male:.2f}, Female: {female:.2f}")

# ===== EDA (10 GRAPHS FIXED) =====
elif menu=="EDA":
    df = load_data()

    color_map={"male":"red","female":"pink"}

    st.plotly_chart(px.histogram(df,x="label",color="label",title="Gender Count",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_pitch",color="label",title="Pitch",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="rms_energy",color="label",title="Energy",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_spectral_centroid",color="label",title="Centroid",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_spectral_bandwidth",color="label",title="Bandwidth",color_discrete_map=color_map))
    st.plotly_chart(px.box(df,x="label",y="mean_pitch",color="label",title="Pitch Box",color_discrete_map=color_map))
    st.plotly_chart(px.box(df,x="label",y="rms_energy",color="label",title="Energy Box",color_discrete_map=color_map))
    st.plotly_chart(px.scatter(df,x="mean_pitch",y="rms_energy",color="label",title="Pitch vs Energy",color_discrete_map=color_map))
    st.plotly_chart(px.imshow(df.corr(),title="Correlation"))
    st.plotly_chart(px.histogram(df,x="mfcc_1_mean",color="label",title="MFCC1",color_discrete_map=color_map))

# ===== CLASSIFICATION (10 GRAPHS + FILTER) =====
elif menu=="Classification":
    df = load_data()

    filt = st.selectbox("Filter",["All","male","female"])
    if filt!="All":
        df=df[df["label"]==filt]

    st.dataframe(df)
    st.download_button("Download CSV",df.to_csv(index=False))

    color_map={"male":"red","female":"pink"}

    st.plotly_chart(px.histogram(df,x="label",color="label",title="Class Dist",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mean_pitch",color="label",title="Pitch",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="rms_energy",color="label",title="Energy",color_discrete_map=color_map))
    st.plotly_chart(px.scatter(df,x="mean_pitch",y="rms_energy",color="label",title="Scatter",color_discrete_map=color_map))
    st.plotly_chart(px.box(df,x="label",y="mean_pitch",color="label",title="Pitch Box",color_discrete_map=color_map))
    st.plotly_chart(px.box(df,x="label",y="rms_energy",color="label",title="Energy Box",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mfcc_1_mean",color="label",title="MFCC1",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mfcc_2_mean",color="label",title="MFCC2",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mfcc_3_mean",color="label",title="MFCC3",color_discrete_map=color_map))
    st.plotly_chart(px.histogram(df,x="mfcc_4_mean",color="label",title="MFCC4",color_discrete_map=color_map))
