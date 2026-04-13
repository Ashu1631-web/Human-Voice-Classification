import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Clustering Classification", layout="wide")

# ================= THEME TOGGLE =================
theme = st.sidebar.toggle("🌗 Dark / Light Mode", value=True)

bg_color = "#0E1117" if theme else "#FFFFFF"
text_color = "#FFFFFF" if theme else "#000000"

# ================= UI (ANIMATED BG + GLASS) =================
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(270deg, #0f2027, #203a43, #2c5364);
    background-size: 600% 600%;
    animation: gradientBG 15s ease infinite;
    color: {text_color};
}}

@keyframes gradientBG {{
    0% {{background-position:0% 50%}}
    50% {{background-position:100% 50%}}
    100% {{background-position:0% 50%}}
}}

.glass {{
    background: rgba(0,0,0,0.6);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
}}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.title("🔐 Human Clustering Classification Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if (u=="admin" and p=="admin123") or (u=="user" and p=="user123"):
            st.session_state.login = True
        else:
            st.error("Invalid credentials ❌")

    st.markdown("</div>", unsafe_allow_html=True)

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
    f.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    f.append(np.mean(librosa.feature.rms(y=y)))
    f.append(np.log(np.sum(y**2)+1e-10))

    pitch = librosa.yin(y, fmin=50, fmax=300)
    f.append(np.mean(pitch))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)
    f.append(np.mean(mfcc))

    return np.array(f).reshape(1,-1)

FEATURE_COLUMNS = ["zero_crossing_rate","rms_energy","log_energy","mean_pitch","mfcc_1_mean"]

# ================= MAIN =================
if not st.session_state.login:
    login()
    st.stop()

menu = st.sidebar.radio("🚀 Navigation", ["Overview","Audio","EDA","Classification","Clustering"])

# ================= OVERVIEW =================
if menu=="Overview":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.title("👥 Human Voice Clustering AI")

    st.markdown("""
### 🧩 Introduction
This project analyzes voice signals using ML to classify gender and cluster similar voices.

### ❗ Problem Statement
- Gender classification  
- Voice clustering  
- Deployable ML system  

### 💡 Solution
- SVM Classification  
- KMeans Clustering  

### 🛠 Tech Stack
Python | pandas | sklearn | Streamlit
""")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= AUDIO =================
elif menu=="Audio":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    file = st.file_uploader("Upload Audio", type=["wav","mp3"])

    if file:
        with open("temp.wav","wb") as f:
            f.write(file.getbuffer())

        st.audio("temp.wav")

        if st.button("Analyze"):
            f = extract_features("temp.wav")
            df = pd.DataFrame(f, columns=FEATURE_COLUMNS)
            f = scaler.transform(df)

            pred = model.predict(f)[0]
            proba = model.predict_proba(f)

            st.success(f"🎯 Prediction: {pred}")
            st.info(f"Confidence: {proba}")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= EDA =================
elif menu=="EDA":
    df = load_data()

    if df is not None:
        st.title("📊 EDA Dashboard")

        st.plotly_chart(px.histogram(df, x="label", color="label"))
        st.plotly_chart(px.imshow(df.corr()))

        for f in ["mean_pitch","zero_crossing_rate","rms_energy","log_energy","mfcc_1_mean"]:
            st.plotly_chart(px.box(df, x="label", y=f, color="label"))

# ================= CLASSIFICATION =================
elif menu=="Classification":
    df = load_data()

    if df is not None:
        st.title("🤖 SVM Classification")

        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.subheader("⚙️ Settings")

        k = st.slider("Features", 5, 43, 5)
        kernel = st.selectbox("Kernel", ["linear","rbf"])
        C = st.slider("C", 0.01,10.0,1.0)

        X = df.drop("label", axis=1)
        y = df["label"]

        selector = SelectKBest(f_classif, k=k)
        X = selector.fit_transform(X,y)

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

        scaler_local = StandardScaler()
        X_train = scaler_local.fit_transform(X_train)
        X_test = scaler_local.transform(X_test)

        clf = SVC(kernel=kernel, C=C, probability=True)
        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_test)

        st.text(classification_report(y_test,y_pred))

        cm = confusion_matrix(y_test,y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        y_prob = clf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test.map({"female":0,"male":1}), y_prob)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr,tpr)
        st.pyplot(fig2)

# ================= CLUSTERING =================
elif menu=="Clustering":
    df = load_data()

    if df is not None:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = df.drop("label", axis=1)
        X = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=2)
        df["cluster"] = kmeans.fit_predict(X)

        st.plotly_chart(px.scatter(df, x="mean_pitch", y="rms_energy", color="cluster"))
