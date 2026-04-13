import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Voice AI", layout="wide")

# ================= LOGIN STATE =================
if "login" not in st.session_state:
    st.session_state.login = False

# ================= LOGIN BACKGROUND ONLY =================
if not st.session_state.login:
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1769525649442-fd8b058b85ab?q=80&w=1074&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= SIDEBAR ANIMATION =================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f2027,#203a43,#2c5364);
}
[data-testid="stSidebar"]::after {
    content:"";
    position:absolute;
    width:100%;
    height:100%;
    background: repeating-linear-gradient(
        90deg,
        rgba(255,255,255,0.05) 0px,
        rgba(255,255,255,0.05) 2px,
        transparent 2px,
        transparent 6px
    );
    animation: move 2s linear infinite;
}
@keyframes move {
    0%{transform:translateY(0);}
    100%{transform:translateY(20px);}
}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if (u=="admin" and p=="admin123") or (u=="user" and p=="user123"):
            st.session_state.login = True
        else:
            st.error("Invalid credentials ❌")

# ================= LOAD =================
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    return model, scaler

model, scaler = load_models()

@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

# ================= FEATURES =================
def extract_features(file):
    import librosa
    y, sr = librosa.load(file, duration=3)

    features = []

    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.rms(y=y)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))

    pitch = librosa.yin(y, fmin=50, fmax=300)
    features.append(np.mean(pitch))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features.append(np.mean(mfcc[i]))

    return np.array(features).reshape(1,-1)

FEATURE_COLUMNS = load_data().drop("label", axis=1).columns

# ================= MAIN =================
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("Menu", ["Overview","EDA","Classification","Clustering","Audio"])

# ================= OVERVIEW =================
if menu=="Overview":
    st.title("🎙️ Human Voice Classification and Clustering")

    st.markdown("""
### 🧩 Introduction  
Voice classification using ML  

### ❗ Problem  
- Gender detection  
- Clustering  

### 💡 Solution  
- SVM  
- KMeans  

### 🛠 Tech  
Python | ML | Streamlit  
""")

# ================= EDA =================
elif menu=="EDA":
    df = load_data()
    df["label"] = df["label"].map({0:"female",1:"male"})

    st.title("📊 EDA Dashboard")

    for i, col in enumerate(df.drop("label", axis=1).columns[:10]):
        st.plotly_chart(px.histogram(
            df,
            x=col,
            color="label",
            title=f"{i+1}. {col} (Male vs Female)"
        ))

# ================= CLASSIFICATION =================
elif menu=="Classification":
    st.title("🧠 Classification")

    df = load_data()
    df["label"] = df["label"].map({0:"female",1:"male"})

    st.dataframe(df.head())

    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    X = df.drop("label", axis=1)
    y = df["label"]

    X_scaled = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

    model_svm = SVC(kernel="rbf", C=5)
    model_svm.fit(X_train, y_train)

    st.success(f"Accuracy: {round(model_svm.score(X_test,y_test)*100,2)}%")

    for i, col in enumerate(X.columns[:10]):
        st.plotly_chart(px.histogram(
            df, x=col, color="label",
            title=f"{i+1}. {col}"
        ))

# ================= CLUSTERING =================
elif menu=="Clustering":
    st.title("🔍 Clustering")

    df = load_data()

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = df.drop("label", axis=1)
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    st.write("Silhouette Score:", round(score,3))

    df["cluster"] = labels

    # 6 graphs
    st.plotly_chart(px.histogram(df, x="cluster", title="1. Cluster Count"))
    st.plotly_chart(px.box(df, x="cluster", y="mean_pitch", title="2. Pitch"))
    st.plotly_chart(px.box(df, x="cluster", y="rms_energy", title="3. Energy"))
    st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="cluster", title="4. Scatter"))
    st.plotly_chart(px.histogram(df, x="cluster", color="label", title="5. Cluster vs Gender"))
    st.plotly_chart(px.scatter(df, x="mean_spectral_centroid", y="mean_spectral_bandwidth", color="cluster", title="6. Spectral"))

# ================= AUDIO =================
elif menu=="Audio":
    file = st.file_uploader("Upload Audio", type=["wav","mp3"])

    if file:
        with open("temp.wav","wb") as f:
            f.write(file.getbuffer())

        st.audio("temp.wav")

        f = extract_features("temp.wav")

        # SAFE FIX
        df_feat = pd.DataFrame(f)
        f_scaled = scaler.transform(df_feat)

        proba = model.predict_proba(f_scaled)
        classes = model.classes_

        result = dict(zip(classes, proba[0]))

        st.success(f"🎯 Prediction: {max(result, key=result.get)}")
        st.info(result)

    st.markdown("### 🎤 Record Voice")
    audio = st.audio_input("Record")

    if audio:
        with open("live.wav","wb") as f:
            f.write(audio.getbuffer())

        st.audio("live.wav")
