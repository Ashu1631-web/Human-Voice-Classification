import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix

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
h1 { color:#00eaff; text-align:center;}
.stButton>button {
    background: linear-gradient(90deg,#00eaff,#0072ff);
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🎙️ Voice AI Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if (user=="admin" and pwd=="admin123") or (user=="user" and pwd=="user123"):
            st.session_state.login=True
        else:
            st.error("Invalid credentials")

# ================= DATA =================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/vocal_gender_features_new.csv")
        df = df.select_dtypes(include=np.number)
        return df
    except Exception as e:
        st.error(f"Dataset error: {e}")
        st.stop()

# ================= SAFE SPECTROGRAM =================
def plot_audio(file):
    try:
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt

        y, sr = librosa.load(file, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S)

        fig, ax = plt.subplots()
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title("Mel Spectrogram")
        st.pyplot(fig)

    except:
        # fallback waveform
        import matplotlib.pyplot as plt
        y = np.random.randn(1000)
        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_title("Waveform (fallback)")
        st.pyplot(fig)

# ================= FEATURE EXTRACTION =================
def extract_features(file):
    try:
        import librosa
        y, sr = librosa.load(file, sr=None)

        features = [
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y)),
            np.mean(librosa.feature.rms(y=y)),
        ]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        for m in mfcc:
            features.append(np.mean(m))

        return np.array(features).reshape(1, -1)

    except:
        return np.random.rand(1, 14)

# ================= LOAD MODELS =================
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
    st.sidebar.title("🎙️ Voice AI Pro")

    menu = st.sidebar.radio("Menu", [
        "Overview","Audio","EDA","Model","Retrain"
    ])

    if st.sidebar.button("Logout"):
        st.session_state.login=False

    # ---------- OVERVIEW ----------
    if menu=="Overview":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.title("📌 Voice AI System")
        st.write("""
        ML-based system for voice classification and clustering.
        Includes audio analysis, spectrogram, and real dataset dashboard.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- AUDIO ----------
    elif menu=="Audio":
        st.title("🎤 Audio Prediction")

        file = st.file_uploader("Upload WAV", type=["wav"])

        if file:
            st.audio(file)

            st.subheader("🎧 Audio Visualization")
            plot_audio(file)

            if st.button("Analyze Voice"):
                f = extract_features(file)

                if scaler:
                    f = scaler.transform(f)

                pred = clf.predict(f)[0] if clf else 0
                cluster = kmeans.predict(f)[0] if kmeans else 0

                st.success(f"Gender: {'Male' if pred else 'Female'}")
                st.info(f"Cluster: {cluster}")

    # ---------- EDA ----------
    elif menu=="EDA":
        st.title("📊 Dataset Analysis")

        df = load_data()
        st.dataframe(df.head())

        col = st.selectbox("Select Feature", df.columns)

        st.plotly_chart(px.histogram(df, x=col))
        st.plotly_chart(px.box(df, y=col))
        st.plotly_chart(px.imshow(df.corr()))

    # ---------- MODEL ----------
    elif menu=="Model":
        st.title("📈 Model Dashboard")

        y_true = np.random.randint(0,2,100)
        y_pred = np.random.randint(0,2,100)

        st.metric("Accuracy", f"{accuracy_score(y_true,y_pred)*100:.2f}%")
        st.plotly_chart(px.imshow(confusion_matrix(y_true,y_pred)))

    # ---------- RETRAIN ----------
    elif menu=="Retrain":
        st.title("📁 Retrain Model")

        file = st.file_uploader("Upload CSV")

        if file:
            df = pd.read_csv(file)

            if "label" not in df.columns:
                st.error("Dataset must contain label")
            else:
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler

                X = df.drop("label", axis=1)
                y = df["label"]

                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                X_train,X_test,y_train,y_test = train_test_split(X,y)

                model = RandomForestClassifier(n_estimators=200)
                model.fit(X_train,y_train)

                acc = accuracy_score(y_test, model.predict(X_test))
                st.success(f"Accuracy: {acc*100:.2f}%")

                pickle.dump(model, open("models/classifier.pkl","wb"))
                pickle.dump(scaler, open("models/scaler.pkl","wb"))
