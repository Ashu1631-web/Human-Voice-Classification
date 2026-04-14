import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import skew, kurtosis
import os

st.set_page_config(page_title="Human Clustering Classification", layout="wide", page_icon="👥")

# ================= GLOBAL CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --pink:   #e8457a;
    --red:    #e03a3a;
    --teal:   #00b89a;
    --dark:   #0a0a12;
    --glass:  rgba(10,10,22,0.82);
    --border: rgba(255,255,255,0.08);
}

.stApp {
    background: var(--dark);
    font-family: 'Inter', system-ui, sans-serif;
}

#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #12001f 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: #e0e0ff !important; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.82rem;
    letter-spacing: 0.02em;
    padding: 6px 0;
}

.glass-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem 2.4rem;
    backdrop-filter: blur(18px);
    margin-bottom: 1.5rem;
}

h1, h2, h3 {
    font-family: 'Inter', system-ui, sans-serif !important;
    letter-spacing: 0.01em;
}

.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1; min-width: 160px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, var(--teal), var(--pink));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .lbl { font-size: 0.78rem; color: #aaa; margin-top: 4px; letter-spacing: 0.04em; }

.login-overlay {
    position: fixed; inset: 0; z-index: 0;
    background: url('https://wallpaperaccess.com/full/9253559.jpg') center/cover no-repeat;
    filter: brightness(0.35) saturate(1.4);
}
.login-card {
    position: relative; z-index: 1;
    max-width: 420px; margin: 8vh auto 0;
    background: rgba(10,10,22,0.88);
    border: 1px solid rgba(0,184,154,0.25);
    border-radius: 22px;
    padding: 2.8rem 2.6rem;
    backdrop-filter: blur(24px);
    box-shadow: 0 0 60px rgba(0,184,154,0.10);
}
.login-title {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.45rem; font-weight: 700;
    color: #fff; text-align: center;
    margin-bottom: 0.3rem;
    letter-spacing: 0.02em;
}
.login-sub {
    text-align: center; color: #888;
    font-size: 0.82rem; margin-bottom: 1.8rem;
}

input[type="text"], input[type="password"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #fff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00c9aa, #0077ff) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 1.6rem !important;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.pred-badge {
    display: inline-block;
    padding: 0.7rem 2rem;
    border-radius: 50px;
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.3rem; font-weight: 600;
    letter-spacing: 0.02em;
    margin: 0.8rem 0;
}
.pred-male   { background: rgba(224,58,58,0.18); border: 2px solid var(--red);  color: var(--red);  }
.pred-female { background: rgba(232,69,122,0.18); border: 2px solid var(--pink); color: var(--pink); }
.pred-unk    { background: rgba(255,200,0,0.15); border: 2px solid #ffc800; color: #ffc800; }

.conf-bar-wrap { margin: 0.6rem 0 1rem; }
.conf-label { font-size: 0.78rem; color: #aaa; margin-bottom: 4px; }
.conf-bar { height: 10px; border-radius: 8px; background: rgba(255,255,255,0.06); overflow: hidden; }
.conf-fill-f { height: 100%; background: linear-gradient(90deg, #e8457a, #f080a8); border-radius: 8px; transition: width 0.6s; }
.conf-fill-m { height: 100%; background: linear-gradient(90deg, #e03a3a, #ff7a00); border-radius: 8px; transition: width 0.6s; }

[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.js-plotly-plot .plotly { border-radius: 14px; }

.sec-divider {
    border: none; border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

.debug-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-top: 1rem;
    font-size: 0.82rem;
    color: #ccc;
    line-height: 1.8;
}
.debug-box b { color: #00b89a; }
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "login" not in st.session_state:
    st.session_state.login = False

# ================= LOGIN PAGE =================
def login_page():
    st.markdown("<div class='login-overlay'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.markdown("<div class='login-title'>👥 Voice AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='login-sub'>Human Clustering & Classification System</div>", unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="admin / user", key="u_inp")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="p_inp")

        if st.button("Login"):
            if (username == "admin" and password == "admin123") or \
               (username == "user"  and password == "user123"):
                st.session_state.login = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try admin/admin123 or user/user123")

        st.markdown("</div>", unsafe_allow_html=True)

# ================= MODEL LOADING =================
@st.cache_resource
def load_models():
    try:
        model  = pickle.load(open("model.pkl",  "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except Exception:
        return None, None

# ================= DATA LOADING =================
@st.cache_data
def load_data():
    if os.path.exists("vocal_gender_features_new.csv"):
        df = pd.read_csv("vocal_gender_features_new.csv")
        if "label" in df.columns:
            unique_vals = df["label"].dropna().unique()
            str_vals = [str(v).strip().lower() for v in unique_vals]
            if all(v in ("0", "1") for v in str_vals):
                df["label"] = df["label"].astype(str).str.strip().map({"0": "Male", "1": "Female"})
        return df
    return None

# ================= FEATURE COLUMNS =================
FEATURE_COLUMNS = [
    'mean_spectral_centroid','std_spectral_centroid',
    'mean_spectral_bandwidth','std_spectral_bandwidth',
    'mean_spectral_contrast','mean_spectral_flatness',
    'mean_spectral_rolloff','zero_crossing_rate','rms_energy',
    'mean_pitch','min_pitch','max_pitch','std_pitch',
    'spectral_skew','spectral_kurtosis','energy_entropy','log_energy',
    'mfcc_1_mean','mfcc_1_std','mfcc_2_mean','mfcc_2_std',
    'mfcc_3_mean','mfcc_3_std','mfcc_4_mean','mfcc_4_std',
    'mfcc_5_mean','mfcc_5_std','mfcc_6_mean','mfcc_6_std',
    'mfcc_7_mean','mfcc_7_std','mfcc_8_mean','mfcc_8_std',
    'mfcc_9_mean','mfcc_9_std','mfcc_10_mean','mfcc_10_std',
    'mfcc_11_mean','mfcc_11_std','mfcc_12_mean','mfcc_12_std',
    'mfcc_13_mean','mfcc_13_std'
]

# ================= AUDIO CONVERSION =================
def convert_to_wav(input_path, output_path="converted_audio.wav"):
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "22050", "-ac", "1", output_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, None
        return None, result.stderr.decode()
    except FileNotFoundError:
        return None, "ffmpeg not found"
    except subprocess.TimeoutExpired:
        return None, "ffmpeg conversion timed out"
    except Exception as e:
        return None, str(e)


def try_read_audio_directly(filepath):
    """ffmpeg ke bina audio directly padhne ki koshish."""
    import librosa

    # Method 1: librosa direct
    try:
        y, sr = librosa.load(filepath, sr=22050, duration=6, mono=True)
        if len(y) > 512:
            return y, sr, None
    except Exception:
        pass

    # Method 2: soundfile
    try:
        import soundfile as sf
        data, sr = sf.read(filepath)
        if data.ndim > 1:
            data = data.mean(axis=1)
        y = data.astype(np.float32)
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        y = y[:22050 * 6]
        if len(y) > 512:
            return y, 22050, None
    except Exception:
        pass

    # Method 3: pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(22050).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(audio.array_type).max
        y = samples[:22050 * 6]
        if len(y) > 512:
            return y, 22050, None
    except Exception:
        pass

    return None, None, "Sab methods fail ho gaye. ffmpeg install karein."


# ================= FEATURE EXTRACTION =================
def extract_features(filepath):
    try:
        import librosa

        y, sr = None, None

        # Step 1: ffmpeg conversion (most reliable)
        converted, err = convert_to_wav(filepath, "converted_audio.wav")
        if converted:
            try:
                y, sr = librosa.load(converted, duration=6, sr=22050)
            except Exception:
                y = None

        # Step 2: fallback — direct read
        if y is None or len(y) < 512:
            y_d, sr_d, err_d = try_read_audio_directly(filepath)
            if y_d is not None and len(y_d) > 512:
                y, sr = y_d, sr_d
            else:
                return None, f"Audio load nahi hua: {err_d or err or 'Unknown'}. ffmpeg install karein."

        if len(y) < 512:
            return None, "Audio bahut chhota hai — kam se kam 2 second record karein."

        feats = []

        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats.extend([np.mean(sc), np.std(sc)])

        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        feats.extend([np.mean(sb), np.std(sb)])

        feats.append(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
        feats.append(np.mean(librosa.feature.spectral_flatness(y=y)))
        feats.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        feats.append(np.mean(librosa.feature.zero_crossing_rate(y)))
        feats.append(np.mean(librosa.feature.rms(y=y)))

        # Wider pitch range — male low pitch (40 Hz) bhi pakde
        pitch = librosa.yin(y, fmin=40, fmax=400)
        # Unvoiced frames (near-zero) hata do
        pitch_voiced = pitch[pitch > 40]
        if len(pitch_voiced) == 0:
            pitch_voiced = pitch
        feats.extend([
            np.mean(pitch_voiced),
            np.min(pitch_voiced),
            np.max(pitch_voiced),
            np.std(pitch_voiced)
        ])

        feats.append(float(skew(y)))
        feats.append(float(kurtosis(y)))
        feats.append(float(-np.sum(y**2 * np.log(y**2 + 1e-10))))
        feats.append(float(np.log(np.sum(y**2) + 1e-10)))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            feats.append(np.mean(mfcc[i]))
            feats.append(np.std(mfcc[i]))

        return np.array(feats).reshape(1, -1), None
    except Exception as e:
        return None, str(e)


# ================= PREDICT HELPER =================
# 60% se kam confidence = "Possibly" dikhao
CONFIDENCE_THRESHOLD = 0.60

def predict_and_display(filepath, model, scaler):
    feats, err = extract_features(filepath)
    if err:
        st.error(f"❌ Feature extraction failed: {err}")
        return

    df_feat = pd.DataFrame(feats, columns=FEATURE_COLUMNS)

    try:
        feats_scaled = scaler.transform(df_feat)
    except Exception as e:
        st.error(f"❌ Scaler error: {e}")
        return

    try:
        proba      = model.predict_proba(feats_scaled)[0]
        prediction = model.predict(feats_scaled)[0]
    except Exception as e:
        st.error(f"❌ Model prediction error: {e}")
        return

    # --- Class index mapping ---
    classes = list(model.classes_)
    female_idx, male_idx = None, None

    for i, c in enumerate(classes):
        s = str(c).strip().lower()
        if s in ("female", "f"):
            female_idx = i
        elif s in ("male", "m"):
            male_idx = i

    if female_idx is None and male_idx is None:
        for i, c in enumerate(classes):
            try:
                val = int(c)
                if val == 0:
                    male_idx = i      # CSV: 0 = Male
                elif val == 1:
                    female_idx = i    # CSV: 1 = Female
            except (ValueError, TypeError):
                pass

    if female_idx is None and male_idx is None:
        female_idx, male_idx = 0, 1

    if female_idx is None:
        female_idx = 1 - male_idx
    if male_idx is None:
        male_idx = 1 - female_idx

    female_prob = float(proba[female_idx])
    male_prob   = float(proba[male_idx])

    # --- Confidence threshold ---
    if female_prob >= CONFIDENCE_THRESHOLD:
        result, badge_cls = "Female 👩", "pred-female"
    elif male_prob >= CONFIDENCE_THRESHOLD:
        result, badge_cls = "Male 👨", "pred-male"
    else:
        # Low confidence — raw prediction se decide karo lekin "Possibly" lagao
        if female_prob > male_prob:
            result, badge_cls = "Possibly Female 👩", "pred-unk"
        else:
            result, badge_cls = "Possibly Male 👨", "pred-unk"

    st.markdown(f"<div class='pred-badge {badge_cls}'>🎯 {result}</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='conf-bar-wrap'>
        <div class='conf-label'>Female Probability — {female_prob*100:.1f}%</div>
        <div class='conf-bar'><div class='conf-fill-f' style='width:{female_prob*100:.1f}%'></div></div>
    </div>
    <div class='conf-bar-wrap'>
        <div class='conf-label'>Male Probability — {male_prob*100:.1f}%</div>
        <div class='conf-bar'><div class='conf-fill-m' style='width:{male_prob*100:.1f}%'></div></div>
    </div>
    """, unsafe_allow_html=True)

    # Debug info — pitch value dikhao
    mean_pitch_val = float(df_feat['mean_pitch'].iloc[0])
    pitch_note = ""
    if mean_pitch_val < 85:
        pitch_note = "⚠️ Pitch bahut low — noise/silence ho sakta hai"
    elif mean_pitch_val <= 180:
        pitch_note = "✅ Typical male pitch range"
    elif mean_pitch_val <= 255:
        pitch_note = "✅ Typical female pitch range"
    else:
        pitch_note = "⚠️ Pitch bahut high — recording check karein"

    st.markdown(f"""
    <div class='debug-box'>
        <b>Debug Info</b><br>
        🎵 <b>Mean Pitch:</b> {mean_pitch_val:.1f} Hz &nbsp;—&nbsp; {pitch_note}<br>
        📊 <b>Female:</b> {female_prob*100:.1f}% &nbsp;|&nbsp;
           <b>Male:</b> {male_prob*100:.1f}% &nbsp;|&nbsp;
           <b>Threshold:</b> {CONFIDENCE_THRESHOLD*100:.0f}%<br>
        💡 <b>Normal range:</b> Male = 85–180 Hz, Female = 165–255 Hz
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 All Extracted Features"):
        st.dataframe(df_feat.T.rename(columns={0: "Value"}), use_container_width=True)


# ================= PLOTLY THEME =================
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e0e0ff",
    font_family="Inter, system-ui, sans-serif",
    title_font_family="Inter, system-ui, sans-serif",
    title_font_size=14,
    margin=dict(l=40, r=20, t=50, b=40),
)

FEMALE_COLOR = "#e8457a"
MALE_COLOR   = "#e03a3a"

def apply_theme(fig):
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


# =====================================================
#  PAGES
# =====================================================

def page_overview():
    st.markdown("<h1 style='color:#fff; font-size:1.8rem;'>👥 Human Voice Clustering AI</h1>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("""
### 📌 Project Overview

An AI-powered pipeline that analyses raw voice audio to perform **gender detection** and **unsupervised voice clustering** — built for real-world scale.
""")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='metric-row'>
    <div class='metric-card'><div class='val'>2</div><div class='lbl'>Classes</div></div>
    <div class='metric-card'><div class='val'>43</div><div class='lbl'>Features</div></div>
    <div class='metric-card'><div class='val'>13</div><div class='lbl'>MFCC Bands</div></div>
    <div class='metric-card'><div class='val'>6s</div><div class='lbl'>Audio Window</div></div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("""
#### 🚀 Features
- 🎤 Upload `.wav` / `.mp3` audio  
- 🔴 Live microphone recording  
- 🧠 Gender prediction with confidence  
- 📊 10-graph EDA dashboard  
- 📋 Classification table view  
- 👥 KMeans voice clustering  
""")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("""
#### 🔄 Workflow
```
Audio File / Mic
      ↓
  Librosa Load (6s)
      ↓
Feature Extraction (43 features)
      ↓
  StandardScaler
      ↓
Trained Classifier
      ↓
Gender Prediction + Confidence
```
""")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("""
#### 🛠 Tech Stack
**Python** · **Streamlit** · **Librosa** · **Scikit-learn** · **Plotly** · **SciPy**

#### 💼 Use Cases
Call-centre analytics · Voice biometrics · AI assistant profiling · Speaker segmentation
""")
    st.markdown("</div>", unsafe_allow_html=True)


def page_audio(model, scaler):
    st.markdown("<h1 style='color:#fff; font-size:1.7rem;'>🎤 Audio Detection</h1>", unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ model.pkl / scaler.pkl not found. Place them in the project directory.")

    # ffmpeg check
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        ffmpeg_ok = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        ffmpeg_ok = False

    if not ffmpeg_ok:
        st.info(
            "ℹ️ **ffmpeg not detected.** Live recording ke liye ffmpeg install karein.\n\n"
            "- **Windows:** `winget install ffmpeg` ya `choco install ffmpeg`\n"
            "- **Mac:** `brew install ffmpeg`\n"
            "- **Linux:** `sudo apt install ffmpeg`\n"
            "- **Streamlit Cloud:** root mein `packages.txt` banao → sirf `ffmpeg` likho"
        )

    tab1, tab2 = st.tabs(["📁 Upload Audio", "🔴 Live Recording"])

    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a `.wav` or `.mp3` file", type=["wav", "mp3"])

        if uploaded:
            ext = os.path.splitext(uploaded.name)[-1].lower() or ".wav"
            save_path = f"temp_upload{ext}"
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())

            st.audio(save_path)
            st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)

            if st.button("🔍 Analyse Audio", key="btn_upload"):
                if model is None:
                    st.error("Model not loaded. Cannot predict.")
                else:
                    with st.spinner("Extracting features…"):
                        predict_and_display(save_path, model, scaler)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "🎙️ Mic button dabao, **5-6 second clearly bolo**, phir stop karo. "
            "Zyada der bolne se accuracy better hoti hai."
        )
        live_audio = st.audio_input("Click to record your voice")

        if live_audio:
            raw_path = "temp_live_raw.webm"
            with open(raw_path, "wb") as f:
                f.write(live_audio.getbuffer())

            st.audio(raw_path)
            st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)

            if model is None:
                st.error("Model not loaded. Cannot predict.")
            else:
                if st.button("🔍 Analyse Recording", key="btn_record"):
                    with st.spinner("Converting & analysing recording…"):
                        converted, err = convert_to_wav(raw_path, "temp_live.wav")
                        if converted:
                            predict_and_display(converted, model, scaler)
                        else:
                            if not ffmpeg_ok:
                                st.warning(
                                    "⚠️ ffmpeg nahi mila — direct analysis try kar rahe hain. "
                                    "Best results ke liye ffmpeg install karein."
                                )
                            else:
                                st.warning(f"⚠️ Conversion issue ({err}) — direct analysis try kar rahe hain…")
                            predict_and_display(raw_path, model, scaler)
        st.markdown("</div>", unsafe_allow_html=True)


def page_eda(df):
    st.markdown("<h1 style='color:#fff; font-size:1.7rem;'>📊 Exploratory Data Analysis</h1>", unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ `vocal_gender_features_new.csv` not found in the project directory.")
        return

    df = df.copy()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower().str.capitalize()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="label", color="label",
                           color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                           title="Class Distribution (Male vs Female)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        fig = px.box(df, x="label", y="mean_pitch", color="label",
                     color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                     title="Mean Pitch by Gender")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="mean_pitch", color="label",
                           color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                           nbins=40, barmode="overlay", opacity=0.75,
                           title="Mean Pitch Distribution")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        fig = px.histogram(df, x="mean_spectral_centroid", color="label",
                           color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                           nbins=40, barmode="overlay", opacity=0.75,
                           title="Spectral Centroid Distribution")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="mean_spectral_bandwidth", color="label",
                           color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                           nbins=40, barmode="overlay", opacity=0.75,
                           title="Spectral Bandwidth Distribution")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        fig = px.histogram(df, x="rms_energy", color="label",
                           color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                           nbins=40, barmode="overlay", opacity=0.75,
                           title="RMS Energy Distribution")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="mfcc_1_mean", color="label",
                           color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                           nbins=40, barmode="overlay", opacity=0.75,
                           title="MFCC-1 Mean Distribution")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        fig = px.histogram(df, x="mfcc_2_mean", color="label",
                           color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                           nbins=40, barmode="overlay", opacity=0.75,
                           title="MFCC-2 Mean Distribution")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="mean_pitch", y="rms_energy", color="label",
                         color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                         opacity=0.7,
                         title="Pitch vs RMS Energy")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        fig = px.scatter(df, x="mean_spectral_centroid", y="mean_spectral_bandwidth",
                         color="label",
                         color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                         opacity=0.7,
                         title="Spectral Centroid vs Bandwidth")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig = px.imshow(corr, color_continuous_scale=["#e03a3a", "#0a0a12", "#e8457a"],
                    title="Feature Correlation Heatmap", aspect="auto")
    fig.update_layout(**PLOT_LAYOUT, height=600)
    st.plotly_chart(fig, use_container_width=True)


def page_classification(df):
    st.markdown("<h1 style='color:#fff; font-size:1.7rem;'>📋 Classification Table & Analysis</h1>", unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ `vocal_gender_features_new.csv` not found.")
        return

    df = df.copy()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower().str.capitalize()

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🗂️ Full Dataset")
    st.markdown(f"**{len(df):,} records · {len(df.columns)} columns**")
    st.dataframe(df, use_container_width=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 📐 Descriptive Statistics")
    st.dataframe(df.describe().T.style.format("{:.4f}"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h2 style='color:#fff; font-size:1.2rem; margin-top:1rem;'>📊 Analysis Charts</h2>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["Gender", "Count"]
        fig = px.bar(counts, x="Gender", y="Count", color="Gender",
                     color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                     title="1. Sample Count per Class")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        num_df = df.select_dtypes(include=np.number)
        means  = df.groupby("label")[num_df.columns.tolist()].mean()
        top_features = means.std().sort_values(ascending=False).head(8).index.tolist()
        fig = px.bar(means[top_features].T.reset_index().melt(id_vars="index"),
                     x="index", y="value", color="label",
                     color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                     barmode="group",
                     title="2. Mean Feature Values by Gender (top 8)")
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        stds = df.groupby("label")[["mean_pitch", "rms_energy", "mean_spectral_centroid",
                                    "mean_spectral_bandwidth", "zero_crossing_rate"]].std().T.reset_index()
        stds = stds.melt(id_vars="index", var_name="Gender", value_name="StdDev")
        fig = px.bar(stds, x="index", y="StdDev", color="Gender",
                     color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                     barmode="group",
                     title="3. Std Deviation of Key Features by Gender")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        fig = px.box(df, x="label", y="mean_spectral_centroid", color="label",
                     color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                     title="4. Spectral Centroid by Gender")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(df, x="label", y="rms_energy", color="label",
                     color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                     title="5. RMS Energy by Gender")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        fig = px.violin(df, x="label", y="zero_crossing_rate", color="label",
                        color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                        box=True,
                        title="6. Zero Crossing Rate by Gender")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        mfcc_cols = [f"mfcc_{i}_mean" for i in range(1, 6)]
        mfcc_means = df.groupby("label")[mfcc_cols].mean().T.reset_index()
        mfcc_means = mfcc_means.melt(id_vars="index", var_name="Gender", value_name="MeanValue")
        fig = px.bar(mfcc_means, x="index", y="MeanValue", color="Gender",
                     color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                     barmode="group",
                     title="7. MFCC 1–5 Mean by Gender")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        fig = px.scatter(df, x="mean_pitch", y="std_pitch", color="label",
                         color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                         opacity=0.65,
                         title="8. Mean Pitch vs Pitch Std Dev")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        pie_data = df["label"].value_counts().reset_index()
        pie_data.columns = ["Gender", "Count"]
        fig = px.pie(pie_data, names="Gender", values="Count",
                     color="Gender",
                     color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                     title="9. Class Balance",
                     hole=0.4)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        fig = px.box(df, x="label", y="mean_spectral_rolloff", color="label",
                     color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                     title="10. Spectral Rolloff by Gender")
        st.plotly_chart(apply_theme(fig), use_container_width=True)


def page_clustering(df):
    st.markdown("<h1 style='color:#fff; font-size:1.7rem;'>👥 Voice Clustering</h1>", unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ `vocal_gender_features_new.csv` not found.")
        return

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler as SK_Scaler

    df = df.copy()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower().str.capitalize()

    num_df = df.select_dtypes(include=np.number)
    X_scaled = SK_Scaler().fit_transform(num_df)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled).astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="cluster", color="cluster",
                           color_discrete_sequence=[FEMALE_COLOR, MALE_COLOR],
                           title="Cluster Size Distribution")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with col2:
        fig = px.scatter(df, x="mean_pitch", y="rms_energy", color="cluster",
                         color_discrete_sequence=[FEMALE_COLOR, MALE_COLOR],
                         opacity=0.7,
                         title="Pitch vs RMS Energy (Clusters)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="cluster", color="label",
                           color_discrete_map={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
                           barmode="stack",
                           title="Cluster vs True Label Overlap")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with col2:
        fig = px.scatter(df, x="mean_spectral_centroid", y="mean_spectral_bandwidth",
                         color="cluster",
                         color_discrete_sequence=[FEMALE_COLOR, MALE_COLOR],
                         opacity=0.7,
                         title="Spectral Centroid vs Bandwidth (Clusters)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 📋 Clustered Dataset")
    st.dataframe(df[["label", "cluster"] + list(num_df.columns[:8])], use_container_width=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
#  ROUTER
# =====================================================
if not st.session_state.login:
    login_page()
    st.stop()

model, scaler = load_models()
df = load_data()

with st.sidebar:
    st.markdown(
        "<p style='text-align:center; font-family:Inter,system-ui,sans-serif; font-size:1rem;"
        " font-weight:600; color:#00b89a; letter-spacing:0.04em; padding: 1rem 0 0.5rem;'>Voice AI</p>",
        unsafe_allow_html=True
    )
    st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:0.5rem 0 1rem;'>",
                unsafe_allow_html=True)

    menu = st.radio("Navigation", ["Overview", "Audio Detection", "EDA", "Classification", "Clustering"])

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:1rem 0 0.5rem;'>",
                unsafe_allow_html=True)
    if st.button("🚪 Logout"):
        st.session_state.login = False
        st.rerun()

if menu == "Overview":
    page_overview()
elif menu == "Audio Detection":
    page_audio(model, scaler)
elif menu == "EDA":
    page_eda(df)
elif menu == "Classification":
    page_classification(df)
elif menu == "Clustering":
    page_clustering(df)
