import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Voice Classification", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR MUSIC =================
st.sidebar.markdown("### 🎵 Background Music")
if os.path.exists("music.mp3"):
    audio_file = open("music.mp3","rb")
    st.sidebar.audio(audio_file.read(), format="audio/mp3")

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if (u=="admin" and p=="admin123") or (u=="user" and p=="user123"):
            st.session_state.login = True
        else:
            st.error("Invalid ❌")

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

# ================= FEATURE =================
def extract_features(file):
    import librosa
    import librosa.effects

    y, sr = librosa.load(file, duration=3)
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)

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

    pitch = librosa.yin(y, fmin=80, fmax=250)
    pitch = pitch[~np.isnan(pitch)]
    if len(pitch) == 0:
        pitch = np.array([0])

    f.extend([np.mean(pitch), np.min(pitch), np.max(pitch), np.std(pitch)])

    f.append(skew(y))
    f.append(kurtosis(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        f.append(np.mean(mfcc[i]))
        f.append(np.std(mfcc[i]))

    return np.array(f).reshape(1, -1)

FEATURE_COLUMNS = [col for col in load_data().columns if col!="label"] if load_data() is not None else []

# ================= MAIN =================
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("Menu", ["Overview","Audio","Recording","EDA","Classification"])

# ================= OVERVIEW =================
if menu=="Overview":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.title("👥 Human Voice AI System")
    st.markdown("""
✔ Gender Detection  
✔ EDA Visualization  
✔ Classification Dashboard  
✔ Audio Upload + Recording  
""")
    st.markdown("</div>", unsafe_allow_html=True)

# ================= AUDIO (UPLOAD - NO CHANGE) =================
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

            # ❗ Upload same logic (unchanged)
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

# ================= RECORDING (FIXED) =================
elif menu=="Recording":
    st.title("🎤 Live Recording Detection")

    audio = st.audio_input("Record")

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

        diff = abs(female_prob - male_prob)

        if diff < 0.15:
            result = "Uncertain ⚠️"
        else:
            result = "Female 👩" if female_prob > male_prob else "Male 👨"

        st.success(f"🎯 Prediction: {result}")
        st.info(f"Confidence → Male: {male_prob:.2f} | Female: {female_prob:.2f}")

# ================= EDA =================
elif menu=="EDA":
    df = load_data()
    if df is not None:
        color_map = {"male":"red","female":"pink"}

        st.plotly_chart(px.histogram(df, x="label", color="label", color_discrete_map=color_map))
        st.plotly_chart(px.histogram(df, x="mean_pitch", color="label", color_discrete_map=color_map))
        st.plotly_chart(px.histogram(df, x="rms_energy", color="label", color_discrete_map=color_map))
        st.plotly_chart(px.histogram(df, x="mean_spectral_centroid", color="label", color_discrete_map=color_map))
        st.plotly_chart(px.histogram(df, x="mean_spectral_bandwidth", color="label", color_discrete_map=color_map))
        st.plotly_chart(px.box(df, x="label", y="mean_pitch", color="label", color_discrete_map=color_map))
        st.plotly_chart(px.box(df, x="label", y="rms_energy", color="label", color_discrete_map=color_map))
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="label", color_discrete_map=color_map))
        st.plotly_chart(px.imshow(df.corr()))
        st.plotly_chart(px.histogram(df, x="mfcc_1_mean", color="label", color_discrete_map=color_map))

# ================= CLASSIFICATION =================
elif menu=="Classification":
    df = load_data()
    if df is not None:
        label = st.selectbox("Filter", ["All","male","female"])

        if label!="All":
            df = df[df["label"]==label]

        st.dataframe(df)

        st.download_button("Download CSV", df.to_csv(index=False), "data.csv")

        color_map = {"male":"red","female":"pink"}

        st.plotly_chart(px.histogram(df, x="label", color="label", color_discrete_map=color_map))
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="label", color_discrete_map=color_map))
