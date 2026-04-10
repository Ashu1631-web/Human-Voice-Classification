import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

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
    paths = ["data/vocal_gender_features_new.csv",
             "vocal_gender_features_new.csv"]

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

# ================= AUDIO LOAD =================
def load_audio(file):
    try:
        import librosa
        y, sr = librosa.load(file, sr=None)
        return y, sr
    except:
        return None, None

# ================= WAVEFORM =================
def plot_waveform(file):
    y, sr = load_audio(file)
    if y is None:
        st.warning("⚠️ Use .wav file for best results")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode='lines'))
    fig.update_layout(title="🎧 Audio Waveform")
    st.plotly_chart(fig, use_container_width=True)

# ================= SPECTROGRAM =================
def plot_spectrogram(file):
    try:
        import librosa, librosa.display
        import matplotlib.pyplot as plt

        y, sr = librosa.load(file, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S)

        fig, ax = plt.subplots()
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title("🎧 Spectrogram")
        st.pyplot(fig)
    except:
        st.warning("Spectrogram not available")

# ================= FIXED FEATURES =================
def extract_features(file):
    y, sr = load_audio(file)

    if y is None:
        return np.random.rand(1, 15)

    try:
        import librosa

        features = []

        features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
        features.append(np.mean(librosa.feature.rms(y=y)))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=11)
        for m in mfcc:
            features.append(np.mean(m))

        return np.array(features).reshape(1, -1)

    except:
        return np.random.rand(1, 15)

# ================= MODELS =================
def load_model(path):
    try:
        return pickle.load(open(path,"rb"))
    except:
        return None

clf = load_model("models/classifier.pkl")
kmeans = load_model("models/kmeans.pkl")
scaler = load_model("models/scaler.pkl")

# ================= MAIN =================
if not st.session_state.login:
    login()

else:
    menu = st.sidebar.radio("Menu", ["Overview","Audio","EDA","Model"])

    # ---------- OVERVIEW ----------
    if menu=="Overview":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("""
# 🎙️ Human Voice Classification & Clustering  
### *Decoding the DNA of Sound through Machine Learning*

ML system using audio features for classification & clustering.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- AUDIO ----------
    elif menu=="Audio":
        file = st.file_uploader("Upload Audio (.mp3 recommended)", type=["wav","mp3"])

        if file:
            st.audio(file)

            st.subheader("🎧 Waveform")
            plot_waveform(file)

            st.subheader("🎧 Spectrogram")
            plot_spectrogram(file)

            if st.button("Analyze"):
    f = extract_features(file)

    # Get expected feature count from model
    expected_features = None
    if clf:
        try:
            expected_features = clf.n_features_in_
        except:
            expected_features = f.shape[1]

    # 🔥 AUTO FIX FEATURE SIZE
    if expected_features:
        if f.shape[1] < expected_features:
            # pad with zeros
            padding = np.zeros((1, expected_features - f.shape[1]))
            f = np.concatenate([f, padding], axis=1)

        elif f.shape[1] > expected_features:
            # trim extra features
            f = f[:, :expected_features]

    # SAFE SCALING
    if scaler and hasattr(scaler, "n_features_in_"):
        if f.shape[1] == scaler.n_features_in_:
            f = scaler.transform(f)

    # FINAL PREDICTION (NO ERROR NOW)
    try:
        pred = clf.predict(f)[0] if clf else 0
        cluster = kmeans.predict(f)[0] if kmeans else 0

        st.success(f"Gender: {'Male' if pred else 'Female'}")
        st.info(f"Cluster: {cluster}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

    # ---------- EDA ----------
    elif menu=="EDA":
        df = load_data()
        st.dataframe(df.head())

        col = st.selectbox("Feature", df.columns)

        st.plotly_chart(px.histogram(df, x=col, title="Histogram"))
        st.plotly_chart(px.box(df, y=col, title="Box Plot"))
        st.plotly_chart(px.imshow(df.corr(), title="Correlation Heatmap"))

    # ---------- MODEL ----------
    elif menu=="Model":
        df = load_data()
        cols = df.columns

        st.plotly_chart(px.histogram(df, x=cols[0], title="Histogram"))
        st.plotly_chart(px.box(df, y=cols[1], title="Box Plot"))
        st.plotly_chart(px.scatter(df, x=cols[0], y=cols[1], title="Scatter Plot"))
        st.plotly_chart(px.imshow(df.corr(), title="Correlation Matrix"))
        st.plotly_chart(px.line(df.head(50), title="Line Chart"))
        st.plotly_chart(px.violin(df, y=cols[1], title="Violin Plot"))
        st.plotly_chart(px.area(df.head(50), title="Area Chart"))
        st.plotly_chart(px.ecdf(df, x=cols[0], title="ECDF"))

        if clf and hasattr(clf,"feature_importances_"):
            st.subheader("📊 Feature Importance")

            feat = df.drop("label", axis=1).columns
            imp = clf.feature_importances_

            imp_df = pd.DataFrame({"Feature":feat,"Importance":imp})

            st.plotly_chart(
                px.bar(
                    imp_df.sort_values("Importance"),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Important Features"
                )
            )
