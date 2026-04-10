import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
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
    import os
    paths = [
        "data/vocal_gender_features_new.csv",
        "vocal_gender_features_new.csv"
    ]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "label" not in df.columns and "gender" in df.columns:
                df.rename(columns={"gender":"label"}, inplace=True)
            return df.select_dtypes(include=np.number)

    st.warning("Upload dataset manually")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        return pd.read_csv(uploaded)

    st.stop()

# ================= AUDIO =================
def extract_features(file):
    import librosa
    y, sr = librosa.load(file, sr=None)

    features = [
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.rms(y=y))
    ]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    for m in mfcc:
        features.append(np.mean(m))

    return np.array(features).reshape(1, -1)

# ================= WAVEFORM =================
def plot_waveform(file):
    try:
        import librosa
        y, sr = librosa.load(file, sr=None)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y, mode='lines'))
        fig.update_layout(title="🎧 Audio Waveform", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Waveform error")

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
        st.pyplot(fig)
    except:
        st.warning("Spectrogram error")

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
    menu = st.sidebar.radio("Menu", [
        "Overview","Audio","EDA","Model","Retrain"
    ])

    # ---------- OVERVIEW ----------
    if menu=="Overview":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("""
# 🎙️ Human Voice Classification & Clustering  
### *Decoding the DNA of Sound through Machine Learning*
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- AUDIO ----------
    elif menu=="Audio":
        file = st.file_uploader("Upload WAV", type=["wav"])

        if file:
            st.audio(file)

            st.subheader("🎧 Waveform")
            plot_waveform(file)

            st.subheader("🎧 Spectrogram")
            plot_spectrogram(file)

            if st.button("Analyze"):
                f = extract_features(file)
                if scaler:
                    f = scaler.transform(f)

                pred = clf.predict(f)[0] if clf else 0
                cluster = kmeans.predict(f)[0] if kmeans else 0

                st.success(f"Gender: {'Male' if pred else 'Female'}")
                st.info(f"Cluster: {cluster}")

    # ---------- EDA ----------
    elif menu=="EDA":
        df = load_data()

        st.dataframe(df.head())

        col = st.selectbox("Feature", df.columns)

        st.plotly_chart(px.histogram(df, x=col))
        st.plotly_chart(px.box(df, y=col))
        st.plotly_chart(px.imshow(df.corr()))

    # ---------- MODEL ----------
    elif menu=="Model":
        df = load_data()

        cols = df.columns

        # 10+ charts
        st.plotly_chart(px.histogram(df, x=cols[0]))
        st.plotly_chart(px.box(df, y=cols[1]))
        st.plotly_chart(px.scatter(df, x=cols[0], y=cols[1]))
        st.plotly_chart(px.imshow(df.corr()))
        st.plotly_chart(px.line(df.head(50)))
        st.plotly_chart(px.density_contour(df, x=cols[0], y=cols[1]))
        st.plotly_chart(px.violin(df, y=cols[1]))
        st.plotly_chart(px.area(df.head(50)))
        st.plotly_chart(px.ecdf(df, x=cols[0]))

        # FEATURE IMPORTANCE
        if clf and hasattr(clf, "feature_importances_"):
            st.subheader("📊 Feature Importance")

            importance = clf.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": df.drop("label", axis=1).columns,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.plotly_chart(px.bar(feat_df.head(10),
                                   x="Importance",
                                   y="Feature",
                                   orientation="h"))

    # ---------- RETRAIN ----------
    elif menu=="Retrain":
        file = st.file_uploader("Upload CSV")

        if file:
            df = pd.read_csv(file)

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
