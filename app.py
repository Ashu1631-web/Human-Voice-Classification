import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Voice AI", layout="wide")

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
    f.extend([np.mean(pitch), np.std(pitch)])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        f.append(np.mean(mfcc[i]))

    return np.array(f).reshape(1, -1)

# ================= MAIN =================
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("Menu", ["Overview","EDA","Classification","Clustering","Audio"])

# ================= OVERVIEW =================
if menu=="Overview":
    st.title("🎙️ Human Voice Classification and Clustering")

    st.markdown("""
- Gender Detection  
- Voice Clustering  
- ML based analysis  
""")

# ================= EDA =================
elif menu=="EDA":
    df = load_data()
    df["label"] = df["label"].map({0:"female",1:"male"})

    st.title("📊 EDA (10 Graphs)")

    features = df.drop("label", axis=1).columns[:10]

    for i, col in enumerate(features):
        st.plotly_chart(px.histogram(
            df, x=col, color="label",
            title=f"{i+1}. {col} Distribution"
        ))

# ================= CLASSIFICATION =================
elif menu=="Classification":
    st.title("🧠 Classification")

    df = load_data()
    df["label"] = df["label"].map({0:"female",1:"male"})

    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import StandardScaler

    st.sidebar.markdown("### ⚙️ Feature Tuning")
    k = st.sidebar.slider("Features", 5, 20, 10)

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

    df_feat = pd.DataFrame({"Feature": selected, "Score": scores})
    st.plotly_chart(px.bar(df_feat, x="Score", y="Feature", orientation='h'))

# ================= CLUSTERING =================
elif menu=="Clustering":
    st.title("🔍 Clustering")

    df = load_data()

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    X = df.drop("label", axis=1)
    X_scaled = StandardScaler().fit_transform(X)

    models = {
        "KMeans": KMeans(n_clusters=2),
        "GMM": GaussianMixture(n_components=2),
        "DBSCAN": DBSCAN()
    }

    results = []

    for name, model in models.items():
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels) if len(set(labels))>1 else -1
        results.append((name, score))

    result_df = pd.DataFrame(results, columns=["Model","Score"])
    st.dataframe(result_df)

    st.plotly_chart(px.bar(result_df, x="Model", y="Score", title="Model Comparison"))

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    st.plotly_chart(px.scatter(
        x=X_pca[:,0], y=X_pca[:,1],
        title="PCA Visualization"
    ))

# ================= AUDIO =================
elif menu=="Audio":
    file = st.file_uploader("Upload Audio", type=["wav","mp3"])

    if file:
        with open("temp.wav","wb") as f:
            f.write(file.getbuffer())

        st.audio("temp.wav")

        f = extract_features("temp.wav")

        # 🔥 FIX (IMPORTANT)
        df_feat = pd.DataFrame(f, columns=scaler.feature_names_in_)

        f_scaled = scaler.transform(df_feat)

        proba = model.predict_proba(f_scaled)

        result = model.classes_[np.argmax(proba)]

        st.success(f"🎯 Prediction: {result}")
        st.info(proba)

    st.markdown("### 🎤 Record Voice")
    audio = st.audio_input("Record")

    if audio:
        with open("live.wav","wb") as f:
            f.write(audio.getbuffer())

        st.audio("live.wav")
