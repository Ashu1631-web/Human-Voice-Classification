import streamlit as st
import pandas as pd
import plotly.express as px
from audio_utils import extract_features
from model import predict_voice
from clustering import cluster_voice
from auth import login, logout

st.set_page_config(page_title="Voice AI System", layout="wide")

# -------- AUTH --------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

login()
if not st.session_state["logged_in"]:
    st.stop()
logout()

# -------- HEADER --------
st.markdown("""
<div style='background: linear-gradient(to right,#1f4037,#99f2c8);
padding:20px;border-radius:12px'>
<h1 style='text-align:center;color:white;'>
🎙️ Human Voice Classification & Clustering
</h1>
<p style='text-align:center;color:white;'>
Analyze voice patterns using ML
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("##")

# -------- UPLOAD --------
file = st.sidebar.file_uploader("Upload Voice (.wav)", type=["wav"])

tab1, tab2, tab3 = st.tabs(["Classification", "Clustering", "Insights"])

# ================= CLASSIFICATION =================
with tab1:
    if file:
        st.audio(file)
        features = extract_features(file)
        result = predict_voice(features)
        st.success(f"Prediction: {result}")

# ================= CLUSTERING =================
with tab2:
    if file:
        features = extract_features(file)
        cluster = cluster_voice(features)
        st.info(f"Cluster Group: {cluster}")

# ================= INSIGHTS =================
with tab3:
    df = pd.read_csv("data/features.csv")

    st.subheader("Cluster Visualization")
    fig = px.scatter(df, x="mfcc1", y="mfcc2", color="label")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance")
    st.bar_chart(df.mean().sort_values(ascending=False).head(10))
