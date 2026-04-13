import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Voice Clustering Classification", layout="wide")

# ================= LOGIN BG ONLY =================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1610733661495-4aa6ed9fc6f4?q=80&w=869&auto=format&fit=crop");
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= LOGIN =================
def login():
    st.title("🔐 Human Voice Clustering Classification Login")
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
    f.extend([np.mean(pitch), np.min(pitch), np.max(pitch), np.std(pitch)])

    f.append(skew(y))
    f.append(kurtosis(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        f.append(np.mean(mfcc[i]))
        f.append(np.std(mfcc[i]))

    return np.array(f).reshape(1, -1)

# ================= MAIN =================
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("Navigate", ["Overview","EDA","Classification","Clustering","Audio"])

# ================= OVERVIEW =================
if menu=="Overview":
    st.title("🎙️ Human Voice Classification and Clustering")

    st.markdown("""
## 🧩 Introduction
Voice classification + clustering using ML.

## ❗ Problem
- Gender detection
- Voice clustering

## 💡 Solution
- SVM Classification
- KMeans Clustering

## 🛠️ Tech
Python | Pandas | Scikit-learn | Streamlit
""")

# ================= EDA =================
elif menu=="EDA":
    df = load_data()
    if df is not None:
        st.title("📊 Exploratory Data Analysis")

        st.markdown("### 1️⃣ Gender Distribution")
        st.plotly_chart(px.histogram(df, x="label", color="label",
                                     color_discrete_map={"male":"blue","female":"pink"}))

        st.markdown("### 2️⃣ Correlation Heatmap")
        st.plotly_chart(px.imshow(df.corr()))

        st.markdown("### 3️⃣ Feature Distribution")
        st.plotly_chart(px.histogram(df, x="mean_pitch", color="label",
                                     barmode="overlay",
                                     color_discrete_map={"male":"blue","female":"pink"}))

        st.plotly_chart(px.histogram(df, x="rms_energy", color="label",
                                     barmode="overlay",
                                     color_discrete_map={"male":"blue","female":"pink"}))

        st.markdown("### 4️⃣ Relationship")
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy",
                                   color="label",
                                   color_discrete_map={"male":"blue","female":"pink"}))

# ================= CLASSIFICATION =================
elif menu=="Classification":
    st.title("🧠 Classification")

    df = load_data()
    if df is not None:
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import confusion_matrix, roc_curve, auc

        # FILTER
        pitch_range = st.sidebar.slider("Pitch Range",
                                       float(df.mean_pitch.min()),
                                       float(df.mean_pitch.max()),
                                       (float(df.mean_pitch.min()),
                                        float(df.mean_pitch.max())))

        df = df[(df.mean_pitch >= pitch_range[0]) & (df.mean_pitch <= pitch_range[1])]

        X = df.drop("label", axis=1)
        y = df["label"]

        X_scaled = StandardScaler().fit_transform(X)

        selector = SelectKBest(f_classif, k=10)
        X_selected = selector.fit_transform(X_scaled, y)

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y)

        model_svm = SVC(kernel="rbf", C=5, probability=True)
        model_svm.fit(X_train, y_train)

        y_pred = model_svm.predict(X_test)

        st.success(f"Accuracy: {round(model_svm.score(X_test,y_test)*100,2)}%")

        # 10 GRAPHS
        for i, col in enumerate(X.columns[:10]):
            st.plotly_chart(px.histogram(df, x=col, color="label",
                                         title=f"{i+1}. {col}",
                                         color_discrete_map={"male":"blue","female":"pink"}))

        # CONFUSION
        cm = confusion_matrix(y_test, y_pred)
        st.plotly_chart(px.imshow(cm, text_auto=True, title="Confusion Matrix"))

        # ROC
        y_prob = model_svm.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label="male")
        roc_auc = auc(fpr, tpr)

        roc_df = pd.DataFrame({"FPR":fpr,"TPR":tpr})
        st.plotly_chart(px.line(roc_df, x="FPR", y="TPR",
                                title=f"ROC Curve (AUC={round(roc_auc,2)})"))

# ================= CLUSTERING =================
elif menu=="Clustering":
    st.title("🔍 Clustering Analysis")

    df = load_data()
    if df is not None:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA

        X = df.drop("label", axis=1)
        X_scaled = StandardScaler().fit_transform(X)

        models = {
            "Agglomerative": AgglomerativeClustering(n_clusters=2),
            "KMeans": KMeans(n_clusters=2),
            "GMM": GaussianMixture(n_components=2),
            "DBSCAN": DBSCAN()
        }

        results = []

        for name, model in models.items():
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels) if len(set(labels))>1 else -1
            results.append((name, score))

        result_df = pd.DataFrame(results, columns=["Model","Silhouette Score"])

        st.dataframe(result_df)

        st.plotly_chart(px.bar(result_df, x="Model", y="Silhouette Score",
                               title="Model Comparison"))

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        kmeans = KMeans(n_clusters=2)
        clusters = kmeans.fit_predict(X_scaled)

        pca_df = pd.DataFrame({
            "PC1": X_pca[:,0],
            "PC2": X_pca[:,1],
            "Cluster": clusters,
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
        df = pd.DataFrame(f)
        f = scaler.transform(df)

        proba = model.predict_proba(f)
        classes = model.classes_

        result = dict(zip(classes, proba[0]))
        st.success(f"Prediction: {max(result, key=result.get)}")
        st.info(result)

    st.markdown("### 🎤 Record Voice")
    audio = st.audio_input("Record")

    if audio:
        with open("live.wav","wb") as f:
            f.write(audio.getbuffer())

        st.audio("live.wav")
