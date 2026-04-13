import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Voice Classification", layout="wide")

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🔐 Human Voice Classification Login")
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
    if os.path.exists("vocal_gender_features_new.csv"):
        return pd.read_csv("vocal_gender_features_new.csv")
    return None

# ================= FEATURE EXTRACTION =================
def extract_features(file):
    import librosa
    y, sr = librosa.load(file, duration=3)

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

    pitch = librosa.yin(y, fmin=50, fmax=300)
    f.extend([np.mean(pitch), np.min(pitch), np.max(pitch), np.std(pitch)])

    f.append(skew(y))
    f.append(kurtosis(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        f.append(np.mean(mfcc[i]))
        f.append(np.std(mfcc[i]))

    return np.array(f).reshape(1, -1)

FEATURE_COLUMNS = [
 'mean_spectral_centroid','std_spectral_centroid','mean_spectral_bandwidth','std_spectral_bandwidth',
 'mean_spectral_contrast','mean_spectral_flatness','mean_spectral_rolloff','zero_crossing_rate',
 'rms_energy','mean_pitch','min_pitch','max_pitch','std_pitch',
 'spectral_skew','spectral_kurtosis','energy_entropy','log_energy',
 'mfcc_1_mean','mfcc_1_std','mfcc_2_mean','mfcc_2_std','mfcc_3_mean','mfcc_3_std',
 'mfcc_4_mean','mfcc_4_std','mfcc_5_mean','mfcc_5_std','mfcc_6_mean','mfcc_6_std',
 'mfcc_7_mean','mfcc_7_std','mfcc_8_mean','mfcc_8_std','mfcc_9_mean','mfcc_9_std',
 'mfcc_10_mean','mfcc_10_std','mfcc_11_mean','mfcc_11_std','mfcc_12_mean','mfcc_12_std',
 'mfcc_13_mean','mfcc_13_std'
]

# ================= MAIN =================
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("Menu", ["Overview","EDA","Classification","Clustering","Audio"])

# ================= OVERVIEW =================
if menu=="Overview":
    st.title("🎙️ Human Voice Classification and Clustering")

    st.markdown("""
## 🧩 Introduction
Analyze voice patterns using ML.

## ❗ Problem Statement
- Gender classification  
- Voice clustering  

## 💡 Solution
- SVM Classification  
- KMeans Clustering  

## 🛠️ Tech Stack
Python | Pandas | Scikit-learn | Streamlit
""")

# ================= EDA =================
elif menu=="EDA":
    df = load_data()

    if df is not None:
        df["label"] = df["label"].map({0:"female",1:"male"})

        st.title("📊 Exploratory Data Analysis")

        # 1️⃣
        st.markdown("### 1️⃣ Gender Distribution")
        st.plotly_chart(px.histogram(df, x="label", color="label"))

        # 2️⃣
        st.markdown("### 2️⃣ Correlation Heatmap")
        st.plotly_chart(px.imshow(df.corr()))

        # 3️⃣
        st.markdown("### 3️⃣ Feature Distribution")
        st.plotly_chart(px.histogram(df, x="mean_pitch", color="label"))
        st.plotly_chart(px.histogram(df, x="rms_energy", color="label"))

        # 4️⃣
        st.markdown("### 4️⃣ Feature Relationship")
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="label"))

# ================= CLASSIFICATION =================
elif menu=="Classification":
    st.title("🧠 Voice Classification")

    df = load_data()

    if df is not None:
        df["label"] = df["label"].map({0:"female",1:"male"})

        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import StandardScaler

        st.sidebar.markdown("### ⚙️ Tuning")
        k = st.sidebar.slider("Select Features", 5, 20, 10)

        X = df.drop("label", axis=1)
        y = df["label"]

        X_scaled = StandardScaler().fit_transform(X)

        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)

        selected = X.columns[selector.get_support()]
        scores = selector.scores_[selector.get_support()]

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y)

        model_svm = SVC(kernel="linear")
        model_svm.fit(X_train, y_train)

        st.success(f"Accuracy: {round(model_svm.score(X_test,y_test)*100,2)}%")

        df_feat = pd.DataFrame({"Feature": selected, "Score": scores}).sort_values("Score")

        st.plotly_chart(px.bar(df_feat, x="Score", y="Feature", orientation='h'))

# ================= CLUSTERING =================
elif menu=="Clustering":
    st.title("🔍 Clustering Analysis")

    df = load_data()

    if df is not None:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA

        X = df.drop("label", axis=1)
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)
        st.write("Silhouette Score:", round(score,3))

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PC1": X_pca[:,0],
            "PC2": X_pca[:,1],
            "Cluster": labels
        })

        st.plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", color="Cluster"))

# ================= AUDIO =================
elif menu=="Audio":
    file = st.file_uploader("Upload Audio", type=["wav","mp3"])

    if file:
        with open("temp.wav","wb") as f:
            f.write(file.getbuffer())

        st.audio("temp.wav")

        f = extract_features("temp.wav")
        df = pd.DataFrame(f, columns=FEATURE_COLUMNS)

        f = scaler.transform(df)
        proba = model.predict_proba(f)

        female_prob = proba[0][0]
        male_prob = proba[0][1]

        if female_prob > 0.6:
            result = "Female 👩"
        elif male_prob > 0.6:
            result = "Male 👨"
        else:
            result = "Uncertain ⚠️"

        st.success(f"🎯 Prediction: {result}")

    st.markdown("### 🎤 Record Voice")
    audio = st.audio_input("Record")

    if audio:
        with open("live.wav","wb") as f:
            f.write(audio.getbuffer())

        st.audio("live.wav")
