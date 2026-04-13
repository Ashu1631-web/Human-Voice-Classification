import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="👥 Human Clustering Classification", layout="wide")

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

menu = st.sidebar.radio("Menu", ["Overview","Audio","EDA","Classification","Clustering"])

# ================= OVERVIEW =================
if menu=="Overview":
    st.title("👥 Human Voice Clustering AI")

    st.markdown("""
### 🧩 Introduction
This project aims to explore how human voice characteristics can be analyzed using machine learning to classify gender and group similar voice patterns using clustering techniques.

### ❗ Problem Statement
There is a need for a lightweight, feature-based ML system that:

- Classifies a voice sample's gender.
- Clusters unlabeled voices into meaningful groups.
- Is easy to deploy in web apps.

### 💡 Proposed Solution
- Classification using SVM  
- Clustering using K-Means, DBSCAN, etc.  

### 🛠️ Technologies Used

| Component | Technology |
|----------|----------|
| Language | Python |
| Data Analysis | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | scikit-learn |
| Interface | Streamlit |
| Deployment | Pickle |
""")

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

            pred = model.predict(f)[0]
            proba = model.predict_proba(f)

            st.success(f"🎯 Prediction: {pred}")
            st.info(f"Confidence: {proba}")

# ================= EDA =================
elif menu=="EDA":
    df = load_data()

    if df is not None:
        st.title("📊 EDA Dashboard")

        # 1. Gender Distribution
        st.subheader("1. Gender Class Distribution")
        st.plotly_chart(px.histogram(df, x="label", color="label"))

        # 2. Correlation Heatmap
        st.subheader("2. Correlation Heatmap")
        st.plotly_chart(px.imshow(df.corr()))

        # 3. Feature Distribution by Gender
        st.subheader("3. Feature Distribution by Gender")

        features = ["mean_pitch","zero_crossing_rate","rms_energy","log_energy","mfcc_1_mean"]

        for f in features:
            st.plotly_chart(px.box(df, x="label", y=f, color="label"))

# ================= CLASSIFICATION =================
elif menu=="Classification":
    df = load_data()

    if df is not None:
        st.title("🤖 Classification (SVM + Feature Selection)")

        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
        import seaborn as sns

        # ⚙️ SETTINGS INLINE (NOT SIDEBAR)
        st.subheader("⚙️ Feature Selection & SVM Tuning")

        k = st.slider("Number of Features to Select", 5, 43, 5)
        kernel = st.selectbox("SVM Kernel", ["linear","rbf"])
        C = st.slider("Regularization (C)", 0.01,10.0,1.0)
        gamma = st.selectbox("Gamma", ["scale","auto"])

        X = df.drop("label", axis=1)
        y = df["label"]

        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(X,y)

        selected_features = X.columns[selector.get_support()]
        st.write("⭐ Top Selected Features:", list(selected_features))

        X_train, X_test, y_train, y_test = train_test_split(X_new,y,test_size=0.2,random_state=42)

        scaler_local = StandardScaler()
        X_train = scaler_local.fit_transform(X_train)
        X_test = scaler_local.transform(X_test)

        clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_test)

        # Report
        st.subheader("📄 Classification Report")
        st.text(classification_report(y_test,y_pred))

        # Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test,y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        # ROC
        st.subheader("📈 ROC Curve")
        y_prob = clf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test.map({"female":0,"male":1}), y_prob)
        roc_auc = auc(fpr,tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax2.legend()
        st.pyplot(fig2)

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
