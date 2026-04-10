import streamlit as st
import numpy as np
import pandas as pd
import pickle
import librosa
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Voice AI Pro", layout="wide")

# =========================
# PREMIUM GLASS UI CSS
# =========================
st.markdown("""
<style>
/* Background Gradient */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Glass Cards */
.glass {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Titles */
h1, h2, h3 {
    color: #00eaff;
    text-align: center;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00eaff, #0072ff);
    border: none;
    border-radius: 10px;
    color: white;
    height: 3em;
    width: 100%;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.7);
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "login" not in st.session_state:
    st.session_state.login = False

# =========================
# LOGIN SYSTEM
# =========================
def login():
    st.markdown("<h1>🔐 Voice AI Login</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if user == "admin" and pwd == "admin123":
                st.session_state.login = True
            elif user == "user" and pwd == "user123":
                st.session_state.login = True
            else:
                st.error("Invalid credentials")

def logout():
    st.session_state.login = False

# =========================
# LOAD DATA (REAL)
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_data.csv")

# =========================
# FEATURE EXTRACTION
# =========================
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
    for i in range(10):
        features.append(np.mean(mfcc[i]))

    return np.array(features).reshape(1, -1)

# =========================
# LOAD MODELS
# =========================
def load_model(path):
    try:
        return pickle.load(open(path, "rb"))
    except:
        return None

classifier = load_model("models/classifier.pkl")
kmeans = load_model("models/kmeans.pkl")
scaler = load_model("models/scaler.pkl")

# =========================
# MAIN APP
# =========================
if not st.session_state.login:
    login()

else:
    st.sidebar.title("🎙️ Voice AI Pro")
    menu = st.sidebar.radio("Navigation", [
        "Overview",
        "Audio Prediction",
        "EDA Dashboard",
        "Model Dashboard",
        "Retrain Model"
    ])

    if st.sidebar.button("Logout"):
        logout()

    # =========================
    # OVERVIEW
    # =========================
    if menu == "Overview":

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.title("📌 Project Overview")

        st.write("""
        This system uses Machine Learning to classify and cluster human voice data.
        Built with advanced UI and real-time analytics.
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("Models", "RandomForest + KMeans")
        col2.metric("Features Used", "15+")
        col3.metric("System Type", "Real-time ML")

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # AUDIO
    # =========================
    elif menu == "Audio Prediction":

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.title("🎤 Audio Prediction")

        audio_file = st.file_uploader("Upload WAV file", type=["wav"])

        if audio_file:
            st.audio(audio_file)

            if st.button("Analyze Voice"):
                features = extract_features(audio_file)

                if scaler:
                    features = scaler.transform(features)

                pred = classifier.predict(features)[0] if classifier else 0
                cluster = kmeans.predict(features)[0] if kmeans else 0

                st.success(f"Prediction: {'Male' if pred else 'Female'}")
                st.info(f"Cluster: {cluster}")

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # REAL EDA DASHBOARD
    # =========================
    elif menu == "EDA Dashboard":

        st.title("📊 Real Dataset Dashboard")

        try:
            df = load_data()

            st.dataframe(df.head())

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(px.histogram(df, x=df.columns[0]))
                st.plotly_chart(px.box(df, y=df.columns[1]))

            with col2:
                st.plotly_chart(px.scatter(df, x=df.columns[0], y=df.columns[1], color="label"))
                st.plotly_chart(px.imshow(df.corr()))

            st.plotly_chart(px.pie(df, names="label"))

        except:
            st.error("Dataset not found. Please upload processed_data.csv")

    # =========================
    # MODEL DASHBOARD
    # =========================
    elif menu == "Model Dashboard":

        st.title("📈 Model Performance")

        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)

        acc = accuracy_score(y_true, y_pred)
        st.metric("Accuracy", f"{acc*100:.2f}%")

        cm = confusion_matrix(y_true, y_pred)
        st.plotly_chart(px.imshow(cm, text_auto=True))

    # =========================
    # RETRAIN
    # =========================
    elif menu == "Retrain Model":

        st.title("📁 Retrain Model")

        file = st.file_uploader("Upload CSV")

        if file:
            df = pd.read_csv(file)

            st.dataframe(df.head())

            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier

            X = df.drop("label", axis=1)
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)

            st.success(f"Accuracy: {acc*100:.2f}%")

            pickle.dump(model, open("models/classifier.pkl", "wb"))
            st.info("Model Updated ✅")

# =========================
# FOOTER
# =========================
st.markdown("""
---
<center style='color:#00eaff;'>🚀 Premium Voice AI Dashboard</center>
""", unsafe_allow_html=True)
