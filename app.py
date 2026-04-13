import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Clustering Classification", layout="wide")

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
        if (u=="admin" and p=="admin123") or (u=="user" and p=="user123"):
            st.session_state.login = True
        else:
            st.error("Invalid credentials ❌")

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
    f.append(-np.sum(y**2 * np.log(y**2 + 1e-10)))
    f.append(np.log(np.sum(y**2) + 1e-10))

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

menu = st.sidebar.radio("Navigate", ["Overview","EDA","Classification","Clustering","Audio"])

# ================= OVERVIEW =================
if menu=="Overview":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.title("🎙️ Human Voice Classification and Clustering")

    st.markdown("""
## 🧩 Introduction
This project analyzes human voice using ML for gender classification and clustering.

## ❗ Problem Statement
- Gender classification  
- Voice clustering  
- Deployable system  

## 💡 Proposed Solution
- SVM Classification  
- KMeans, DBSCAN Clustering  

## 🛠️ Technologies
Python | Pandas | NumPy | Scikit-learn | Streamlit
""")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= EDA =================
elif menu=="EDA":
    df = load_data()

    if df is not None:
        st.title("📊 Exploratory Data Analysis")

        st.markdown("### 1️⃣ Gender Distribution")
        st.plotly_chart(px.histogram(df, x="label"))

        st.markdown("### 2️⃣ Correlation Heatmap")
        st.plotly_chart(px.imshow(df.corr()))

        st.markdown("### 3️⃣ Feature Distributions")
        st.plotly_chart(px.histogram(df, x="mean_pitch", color="label"))
        st.plotly_chart(px.histogram(df, x="rms_energy", color="label"))

        st.markdown("### 4️⃣ Relationship")
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="label"))

# ================= CLASSIFICATION =================
elif menu=="Classification":
    st.title("🧠 SVM Classification")

    df = load_data()
    if df is not None:
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import StandardScaler

        X = df.drop("label", axis=1)
        y = df["label"]

        X_scaled = StandardScaler().fit_transform(X)

        k = st.sidebar.slider("Features", 5, 20, 10)

        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)

        selected = X.columns[selector.get_support()]
        scores = selector.scores_[selector.get_support()]

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y)

        model_svm = SVC(kernel="linear")
        model_svm.fit(X_train, y_train)

        acc = model_svm.score(X_test, y_test)
        st.success(f"Accuracy: {round(acc*100,2)}%")

        df_feat = pd.DataFrame({"Feature": selected, "Score": scores}).sort_values("Score")

        st.plotly_chart(px.bar(df_feat, x="Score", y="Feature", orientation='h'))

# ================= CLUSTERING =================
elif menu=="Clustering":
    st.title("🔍 Clustering")

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

        st.write("Silhouette Score:", silhouette_score(X_scaled, labels))

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PC1": X_pca[:,0],
            "PC2": X_pca[:,1],
            "Cluster": labels,
            "Actual": df["label"]
        })

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", color="Cluster"))

        with col2:
            st.plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", color="Actual"))

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
        classes = model.classes_
        proba_dict = dict(zip(classes, proba[0]))

        result = max(proba_dict, key=proba_dict.get)

        st.success(f"Prediction: {result}")
        st.info(proba_dict)

    st.markdown("### 🎤 Record Voice")
    audio = st.audio_input("Record")

    if audio:
        with open("live.wav","wb") as f:
            f.write(audio.getbuffer())

        st.audio("live.wav")
