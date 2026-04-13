import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Voice AI", layout="wide")

# ================= UI =================
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1610733661495-4aa6ed9fc6f4?q=80&w=869&auto=format&fit=crop");
    background-size: cover;
}
.glass {
    background: rgba(0,0,0,0.75);
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if (u=="admin" and p=="admin123"):
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

# ================= AUDIO CLEAN =================
def clean_audio(y):
    return y / (np.max(np.abs(y)) + 1e-9)

# ================= FEATURES =================
def extract_features(file):
    import librosa

    y, sr = librosa.load(file, duration=5)
    y = clean_audio(y)

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

    pitch = librosa.yin(y, fmin=50, fmax=500)
    f.extend([np.mean(pitch), np.min(pitch), np.max(pitch), np.std(pitch)])

    f.append(skew(y))
    f.append(kurtosis(y))
    f.append(np.log(np.sum(y**2)+1e-10))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        f.append(np.mean(mfcc[i]))
        f.append(np.std(mfcc[i]))

    return np.array(f).reshape(1, -1)

# ================= PREDICT =================
def predict(audio_path):
    f = extract_features(audio_path)

    # auto columns
    df = pd.DataFrame(f)
    df.columns = [f"f{i}" for i in range(df.shape[1])]

    f_scaled = scaler.transform(df.values)

    proba = model.predict_proba(f_scaled)
    classes = list(model.classes_)

    classes_lower = [str(c).lower() for c in classes]

    female_index = 0
    male_index = 1

    if "female" in classes_lower:
        female_index = classes_lower.index("female")
    elif "f" in classes_lower:
        female_index = classes_lower.index("f")

    if "male" in classes_lower:
        male_index = classes_lower.index("male")
    elif "m" in classes_lower:
        male_index = classes_lower.index("m")

    female_prob = proba[0][female_index]
    male_prob = proba[0][male_index]

    result = "Female 👩" if female_prob > male_prob else "Male 👨"

    return result, female_prob, male_prob

# ================= MAIN =================
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("Menu", ["Overview","Audio","EDA","Model","Clustering"])

# ================= OVERVIEW =================
if menu=="Overview":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.title("👥 Human Voice AI System")
    st.markdown("Audio → Features → Model → Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

# ================= AUDIO =================
elif menu=="Audio":

    file = st.file_uploader("Upload Audio", type=["wav","mp3"])

    if file:
        with open("temp.wav","wb") as f:
            f.write(file.getbuffer())

        st.audio("temp.wav")

        if st.button("Analyze"):
            res, f_prob, m_prob = predict("temp.wav")
            st.success(f"🎯 {res}")
            st.info(f"Female: {f_prob:.2f} | Male: {m_prob:.2f}")

    st.markdown("### 🎤 Record Voice")
    audio = st.audio_input("Record")

    if audio:
        with open("live.wav","wb") as f:
            f.write(audio.getbuffer())

        st.audio("live.wav")
        res, f_prob, m_prob = predict("live.wav")
        st.success(f"🎯 {res}")
        st.info(f"Female: {f_prob:.2f} | Male: {m_prob:.2f}")

# ================= EDA =================
elif menu=="EDA":
    df = load_data()
    if df is not None:
        st.title("📊 Analysis")
        st.plotly_chart(px.histogram(df, x="label"))
        st.plotly_chart(px.histogram(df, x="mean_pitch"))
        st.plotly_chart(px.box(df, x="label", y="mean_pitch"))
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="label"))
        st.plotly_chart(px.imshow(df.corr()))

# ================= MODEL =================
elif menu=="Model":
    df = load_data()
    if df is not None:
        from sklearn.metrics import confusion_matrix

        X = df.drop("label", axis=1)
        y = df["label"]

        preds = model.predict(scaler.transform(X))
        cm = confusion_matrix(y, preds)

        fig = ff.create_annotated_heatmap(
            z=cm,
            x=["Female","Male"],
            y=["Female","Male"]
        )
        st.plotly_chart(fig)

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

        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="cluster"))
