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
    background: rgba(0,0,0,0.7);
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🔐 Human Clustering Classification Login")
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
    f.append(-np.sum(y**2 * np.log(y**2 + 1e-10)))
    f.append(np.log(np.sum(y**2) + 1e-10))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        f.append(np.mean(mfcc[i]))
        f.append(np.std(mfcc[i]))

    return np.array(f).reshape(1, -1)

FEATURE_COLUMNS = [...]  # same as your code (unchanged)

# ================= MAIN =================
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("Menu", ["Overview","Audio","EDA","Model","Clustering"])

# ================= OVERVIEW =================
if menu=="Overview":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.title("👥 Human Voice Clustering AI")
    st.markdown("AI Voice Analysis System with EDA + ML + Clustering 🚀")
    st.markdown("</div>", unsafe_allow_html=True)

# ================= AUDIO =================
elif menu=="Audio":
    file = st.file_uploader("Upload Audio", type=["wav","mp3"])

    if file:
        with open("temp.wav","wb") as f:
            f.write(file.getbuffer())

        st.audio("temp.wav")

        if st.button("Analyze"):
            f = extract_features("temp.wav")
            df = pd.DataFrame(f, columns=FEATURE_COLUMNS)
            f = scaler.transform(df)

            proba = model.predict_proba(f)
            st.success(f"Prediction: {model.predict(f)[0]}")
            st.info(f"Confidence: {proba}")

# ================= EDA =================
elif menu=="EDA":
    df = load_data()

    if df is not None:
        st.title("📊 Advanced EDA Dashboard")

        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        st.write("Missing Values:", df.isnull().sum())
        st.write(df.describe())

        st.plotly_chart(px.histogram(df, x="label"))

        feature = st.selectbox("Select Feature", df.columns)
        st.plotly_chart(px.histogram(df, x=feature))
        st.plotly_chart(px.box(df, x="label", y=feature))

        st.plotly_chart(px.imshow(df.corr()))
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="label"))

# ================= MODEL =================
elif menu=="Model":
    df = load_data()

    if df is not None:
        st.title("🤖 Model Insights")

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix
        from sklearn.ensemble import RandomForestClassifier
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler

        X = df.drop("label", axis=1)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        scaler_local = StandardScaler()
        X_train = scaler_local.fit_transform(X_train)
        X_test = scaler_local.transform(X_test)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        importance = pd.Series(clf.feature_importances_, index=X.columns)
        st.bar_chart(importance.sort_values(ascending=False).head(10))

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

        st.plotly_chart(px.histogram(df, x="cluster"))
        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="cluster"))
        st.plotly_chart(px.histogram(df, x="cluster", color="label"))
