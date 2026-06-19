"""
app.py - SarcasmIQ: AI-Powered Sarcasm Detection
Professional Streamlit UI - Cloud Deployment Version
"""

import re, pickle, os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SarcasmIQ — AI Sarcasm Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

STOPWORDS, LEMMATIZER = setup_nltk()

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0a0f; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #0a0a12 100%);
    border-right: 1px solid rgba(139,92,246,0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSelectbox"] > div > div {
    background: rgba(139,92,246,0.1) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stTextArea textarea {
    background: rgba(15,15,30,0.8) !important;
    border: 2px solid rgba(139,92,246,0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(139,92,246,0.8) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}
.stTextArea textarea::placeholder { color: #4a5568 !important; }
.stButton > button {
    width: 100%; border-radius: 12px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    border: none !important; padding: 0.75rem 1.5rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,15,30,0.5);
    border-radius: 12px; padding: 4px; gap: 4px;
    border: 1px solid rgba(139,92,246,0.2);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; color: #64748b !important;
    font-weight: 600 !important; padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    color: white !important;
}
[data-testid="metric-container"] {
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px; padding: 1rem;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-size:0.8rem !important; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #e2e8f0 !important; font-size:1.4rem !important; font-weight:700 !important; }
hr { border-color: rgba(139,92,246,0.2) !important; }
.streamlit-expanderHeader {
    background: rgba(139,92,246,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important; font-weight: 600 !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text_ml(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join([LEMMATIZER.lemmatize(w) for w in text.split()
                     if w not in STOPWORDS and len(w) > 1])

def clean_text_dl(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    try:
        tfidf = pickle.load(open("models/tfidf_vectorizer.pkl","rb"))
        model = pickle.load(open("models/sarcasm_model.pkl","rb"))
        meta  = pickle.load(open("models/model_metadata.pkl","rb"))
        return tfidf, model, meta, True
    except Exception as e:
        return None, None, {}, False

@st.cache_resource
def load_dl_model():
    try:
        import tensorflow as tf, os
        saved_model_path = "models/saved_model"
        h5_path          = "models/best_dl_model.h5"
        keras_path       = "models/best_dl_model.keras"
        if os.path.isdir(saved_model_path):
            model = tf.saved_model.load(saved_model_path)
        elif os.path.exists(h5_path) and os.path.getsize(h5_path) > 10_000:
            model = tf.keras.models.load_model(h5_path, compile=False)
        elif os.path.exists(keras_path) and os.path.getsize(keras_path) > 10_000:
            model = tf.keras.models.load_model(keras_path, compile=False)
        else:
            return None, None, {}, False
        tokenizer = pickle.load(open("models/dl_tokenizer.pkl", "rb"))
        meta      = pickle.load(open("models/dl_metadata.pkl", "rb"))
        return model, tokenizer, meta, True
    except Exception:
        return None, None, {}, False

tfidf, ml_model, ml_meta, ML_OK   = load_ml_model()
dl_model, dl_tokenizer, dl_meta, DL_OK = load_dl_model()

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict_ml(text):
    vec = tfidf.transform([clean_text_ml(text)])
    p   = ml_model.predict_proba(vec)[0]
    return float(p[1]), float(p[0])

def predict_dl(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq    = dl_tokenizer.texts_to_sequences([clean_text_dl(text)])
    padded = pad_sequences(seq, maxlen=dl_meta.get("max_len",40),
                           padding="post", truncating="post")
    import tensorflow as tf, numpy as np
    inp = tf.constant(padded, dtype=tf.float32)
    if hasattr(dl_model, 'predict'):
        raw = dl_model.predict(padded, verbose=0)
    else:
        raw = dl_model.serve(inp).numpy()
    p = float(raw[0][0])
    return p, 1-p

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d0d1a 0%, #130d2e 50%, #0d0d1a 100%);
    border-bottom: 1px solid rgba(139,92,246,0.3);
    padding: 1.5rem 2rem 1.2rem 2rem;
    margin: -2rem -2rem 2rem -2rem;
    position: relative; overflow: hidden;
">
  <div style="position:absolute;top:-60px;left:10%;width:300px;height:300px;
              background:radial-gradient(circle,rgba(124,58,237,0.15),transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;top:-40px;right:15%;width:250px;height:250px;
              background:radial-gradient(circle,rgba(79,70,229,0.12),transparent 70%);pointer-events:none;"></div>
  <div style="display:flex;align-items:center;justify-content:space-between;position:relative;">
    <div style="display:flex;align-items:center;gap:1rem;">
      <div style="background:linear-gradient(135deg,#7c3aed,#4f46e5);border-radius:14px;
                  width:48px;height:48px;display:flex;align-items:center;justify-content:center;
                  font-size:1.6rem;box-shadow:0 4px 20px rgba(124,58,237,0.5);">🎭</div>
      <div>
        <div style="font-size:1.6rem;font-weight:900;
                    background:linear-gradient(90deg,#a78bfa,#818cf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">SarcasmIQ</div>
        <div style="font-size:0.75rem;color:#64748b;font-weight:500;
                    letter-spacing:1px;text-transform:uppercase;">AI-Powered Sarcasm Detection Engine</div>
      </div>
    </div>
    <div style="display:flex;gap:0.6rem;flex-wrap:wrap;">
      <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.4);
                   color:#a78bfa;border-radius:999px;padding:4px 14px;font-size:0.75rem;font-weight:600;">
        🧠 Classical ML · 80.5%
      </span>
      <span style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.4);
                   color:#34d399;border-radius:999px;padding:4px 14px;font-size:0.75rem;font-weight:600;">
        🔬 GloVe + BiLSTM · 88.9%
      </span>
      <span style="background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.4);
                   color:#fbbf24;border-radius:999px;padding:4px 14px;font-size:0.75rem;font-weight:600;">
        NLP Capstone Project 2
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-size:0.7rem;color:#4a5568;letter-spacing:2px;
                text-transform:uppercase;font-weight:700;margin-bottom:0.8rem;">⚙️ Configuration</div>
    """, unsafe_allow_html=True)

    model_options = []
    if ML_OK: model_options.append("🧠 Classical ML  (80.5%)")
    if DL_OK: model_options.append("🔬 GloVe + Stacked BiLSTM  (88.9%)")
    if not model_options:
        st.error("No models found!")
        st.stop()

    selected  = st.selectbox("Active Model", model_options)
    threshold = st.slider("Sarcasm Threshold", 0.0, 1.0, 0.5, 0.01)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Correct metrics per selected model ──
    if "Classical" in selected and ML_OK:
        acc   = ml_meta.get("test_accuracy", 0.805)
        f1    = ml_meta.get("test_f1", 0.794)
        auc   = ml_meta.get("test_auc", 0.889)
        mname = ml_meta.get("model_name", "Stacking Ensemble")
    elif DL_OK:
        acc   = dl_meta.get("test_accuracy", 0.889)
        f1    = dl_meta.get("test_f1", 0.883)
        auc   = dl_meta.get("test_auc", 0.957)
        mname = dl_meta.get("model_name", "GloVe + Stacked BiLSTM")
    else:
        acc, f1, auc, mname = 0, 0, 0, "N/A"

    st.markdown(f"""
    <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);
                border-radius:12px;padding:1rem;">
      <div style="font-size:0.7rem;color:#64748b;letter-spacing:1px;text-transform:uppercase;
                  font-weight:700;margin-bottom:0.8rem;">📊 Model Performance</div>
      <div style="font-size:0.8rem;color:#a78bfa;font-weight:600;margin-bottom:0.8rem;">{mname}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;">
        <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:0.6rem;text-align:center;">
          <div style="font-size:1.1rem;font-weight:800;color:#a78bfa;">{acc:.1%}</div>
          <div style="font-size:0.7rem;color:#64748b;">Accuracy</div>
        </div>
        <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:0.6rem;text-align:center;">
          <div style="font-size:1.1rem;font-weight:800;color:#34d399;">{f1:.1%}</div>
          <div style="font-size:0.7rem;color:#64748b;">F1 Score</div>
        </div>
        <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:0.6rem;
                    text-align:center;grid-column:span 2;">
          <div style="font-size:1.1rem;font-weight:800;color:#fbbf24;">{auc:.1%}</div>
          <div style="font-size:0.7rem;color:#64748b;">ROC-AUC</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.7rem;color:#4a5568;letter-spacing:2px;
                text-transform:uppercase;font-weight:700;margin-bottom:0.8rem;">🚀 Accuracy Journey</div>
    """, unsafe_allow_html=True)

    journey = [
        ("TF-IDF + LR",        0.788, "#4a5568"),
        ("TF-IDF + Ensemble",  0.805, "#6366f1"),
        ("BiLSTM (random)",    0.798, "#4a5568"),
        ("GloVe + BiLSTM",     0.865, "#8b5cf6"),
        ("GloVe + BiGRU",      0.877, "#a78bfa"),
        ("GloVe + Stacked ✅", 0.889, "#34d399"),
    ]
    for name, val, color in journey:
        bar_w = int(val * 130)
        st.markdown(f"""
        <div style="margin-bottom:0.5rem;">
          <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
            <span style="font-size:0.72rem;color:#94a3b8;">{name}</span>
            <span style="font-size:0.72rem;color:{color};font-weight:700;">{int(val*100)}%</span>
          </div>
          <div style="background:rgba(255,255,255,0.05);border-radius:999px;height:5px;">
            <div style="width:{bar_w}px;max-width:100%;background:{color};
                        border-radius:999px;height:5px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if not DL_OK:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
                    border-radius:8px;padding:0.7rem;margin-top:1rem;">
          <div style="font-size:0.75rem;color:#fbbf24;">
            ⚠️ GloVe model loading — using Classical ML mode
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Single Prediction","📁  Batch Analysis","📊  Model Insights"])

# ══ TAB 1 ══
with tab1:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown("""
        <div style="margin-bottom:1rem;">
          <div style="font-size:1.4rem;font-weight:800;color:#e2e8f0;margin-bottom:0.3rem;">
            Analyze a Headline
          </div>
          <div style="font-size:0.875rem;color:#64748b;">
            Enter any news headline or sentence to detect sarcasm using AI
          </div>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            "headline", height=130, label_visibility="collapsed",
            placeholder="e.g. 'Scientists discover water is, in fact, wet' ...",
        )

        c1, c2 = st.columns([2,1])
        predict_btn = c1.button("🔮  Detect Sarcasm", type="primary", use_container_width=True)
        c2.button("✕  Clear", type="secondary", use_container_width=True)

        st.markdown("""
        <div style="font-size:0.75rem;color:#4a5568;font-weight:600;
                    text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem 0;">
          Try a sample
        </div>""", unsafe_allow_html=True)

        samples = [
            ("🎭", "Scientists discover water is, in fact, wet"),
            ("🎭", "Area man passionate defender of what he calls 'my team'"),
            ("📰", "Stock market hits record high amid strong earnings"),
            ("📰", "Local firefighters rescue family from burning building"),
        ]
        for emoji, text in samples:
            if st.button(f"{emoji}  {text}", key=text, use_container_width=True):
                user_input  = text
                predict_btn = True

    with col_right:
        if predict_btn and user_input and user_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    if "Classical" in selected and ML_OK:
                        sarc_p, not_p = predict_ml(user_input)
                    elif DL_OK:
                        sarc_p, not_p = predict_dl(user_input)
                    else:
                        st.error("No model available.")
                        st.stop()

                    is_sarc = sarc_p >= threshold
                    conf    = sarc_p if is_sarc else not_p

                    if is_sarc:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#7c1d1d,#991b1b,#7c1d1d);
                                    border:1px solid rgba(239,68,68,0.5);border-radius:20px;
                                    padding:2rem;text-align:center;
                                    box-shadow:0 0 40px rgba(239,68,68,0.2);">
                          <div style="font-size:3.5rem;margin-bottom:0.5rem;">😏</div>
                          <div style="font-size:1.8rem;font-weight:900;color:#fca5a5;
                                      letter-spacing:-0.5px;margin-bottom:0.3rem;">Sarcastic!</div>
                          <div style="font-size:0.85rem;color:#f87171;margin-bottom:1.2rem;">
                            This text carries sarcastic intent
                          </div>
                          <div style="background:rgba(0,0,0,0.3);border-radius:12px;
                                      padding:0.8rem;display:inline-block;min-width:160px;">
                            <div style="font-size:2rem;font-weight:900;color:white;">{conf:.1%}</div>
                            <div style="font-size:0.75rem;color:#fca5a5;font-weight:600;">CONFIDENCE</div>
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#064e3b,#065f46,#064e3b);
                                    border:1px solid rgba(16,185,129,0.5);border-radius:20px;
                                    padding:2rem;text-align:center;
                                    box-shadow:0 0 40px rgba(16,185,129,0.2);">
                          <div style="font-size:3.5rem;margin-bottom:0.5rem;">🙂</div>
                          <div style="font-size:1.8rem;font-weight:900;color:#6ee7b7;
                                      letter-spacing:-0.5px;margin-bottom:0.3rem;">Not Sarcastic</div>
                          <div style="font-size:0.85rem;color:#34d399;margin-bottom:1.2rem;">
                            This text appears genuine
                          </div>
                          <div style="background:rgba(0,0,0,0.3);border-radius:12px;
                                      padding:0.8rem;display:inline-block;min-width:160px;">
                            <div style="font-size:2rem;font-weight:900;color:white;">{conf:.1%}</div>
                            <div style="font-size:0.75rem;color:#6ee7b7;font-weight:600;">CONFIDENCE</div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    st.write("")
                    st.markdown("""<div style="font-size:0.75rem;color:#4a5568;font-weight:700;
                                text-transform:uppercase;letter-spacing:1px;margin-bottom:0.6rem;">
                                Probability Breakdown</div>""", unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    m1.metric("🙂 Not Sarcastic", f"{not_p:.1%}")
                    m2.metric("😏 Sarcastic",     f"{sarc_p:.1%}")

                    st.markdown(f"""
                    <div style="margin:0.8rem 0;">
                      <div style="background:rgba(255,255,255,0.05);border-radius:999px;
                                  height:10px;overflow:hidden;position:relative;">
                        <div style="position:absolute;left:0;top:0;height:100%;
                                    width:{not_p*100:.1f}%;
                                    background:linear-gradient(90deg,#10b981,#34d399);
                                    border-radius:999px 0 0 999px;"></div>
                        <div style="position:absolute;right:0;top:0;height:100%;
                                    width:{sarc_p*100:.1f}%;
                                    background:linear-gradient(90deg,#f87171,#ef4444);
                                    border-radius:0 999px 999px 0;"></div>
                      </div>
                      <div style="display:flex;justify-content:space-between;margin-top:4px;">
                        <span style="font-size:0.72rem;color:#34d399;">← Not Sarcastic</span>
                        <span style="font-size:0.72rem;color:#ef4444;">Sarcastic →</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    if ML_OK and DL_OK:
                        with st.expander("🔁 Compare across both models"):
                            rows = []
                            p_ml, _ = predict_ml(user_input)
                            p_dl, _ = predict_dl(user_input)
                            for mname, prob in [("Classical ML (80.5%)", p_ml),
                                                ("GloVe + Stacked BiLSTM (88.9%)", p_dl)]:
                                rows.append({
                                    "Model":        mname,
                                    "Sarcasm %":    f"{prob:.1%}",
                                    "Verdict":      "😏 Sarcastic" if prob>=threshold else "🙂 Not Sarcastic",
                                    "Confidence":   f"{max(prob,1-prob):.1%}",
                                })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        else:
            st.markdown("""
            <div style="border:2px dashed rgba(139,92,246,0.2);border-radius:20px;
                        padding:3rem;text-align:center;background:rgba(139,92,246,0.03);">
              <div style="font-size:3rem;margin-bottom:1rem;opacity:0.4;">🎭</div>
              <div style="font-size:1rem;color:#4a5568;font-weight:600;">Your result will appear here</div>
              <div style="font-size:0.8rem;color:#374151;margin-top:0.3rem;">
                Enter a headline and click Detect Sarcasm
              </div>
            </div>""", unsafe_allow_html=True)

# ══ TAB 2 ══
with tab2:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-size:1.4rem;font-weight:800;color:#e2e8f0;margin-bottom:0.3rem;">Batch Analysis</div>
      <div style="font-size:0.875rem;color:#64748b;">Upload a CSV to analyze hundreds of headlines at once</div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.dataframe(df_batch.head(), use_container_width=True, hide_index=True)
        col_choice = st.selectbox("Select text column", df_batch.columns)

        if st.button("⚡ Run Batch Analysis", type="primary"):
            prog = st.progress(0, text="Analyzing...")
            probs = []
            for i, text in enumerate(df_batch[col_choice].astype(str)):
                try:
                    if "Classical" in selected and ML_OK:
                        p, _ = predict_ml(text)
                    elif DL_OK:
                        p, _ = predict_dl(text)
                    else: p = 0.5
                    probs.append(p)
                except: probs.append(0.5)
                if i % 10 == 0:
                    prog.progress((i+1)/len(df_batch), text=f"Processing {i+1}/{len(df_batch)}...")

            prog.empty()
            df_batch["sarcasm_probability"] = probs
            df_batch["prediction"] = ["😏 Sarcastic" if p>=threshold else "🙂 Not Sarcastic" for p in probs]
            df_batch["confidence"] = [f"{max(p,1-p):.1%}" for p in probs]

            n_s = sum(1 for p in probs if p >= threshold)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total", f"{len(df_batch):,}")
            m2.metric("😏 Sarcastic", f"{n_s:,}", f"{n_s/len(probs):.1%}")
            m3.metric("🙂 Not Sarcastic", f"{len(probs)-n_s:,}")
            m4.metric("Avg Confidence", f"{np.mean([max(p,1-p) for p in probs]):.1%}")

            st.dataframe(df_batch, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Results", df_batch.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv", use_container_width=True)
    else:
        st.markdown("""
        <div style="border:2px dashed rgba(139,92,246,0.2);border-radius:20px;
                    padding:4rem;text-align:center;">
          <div style="font-size:3rem;margin-bottom:1rem;opacity:0.3;">📁</div>
          <div style="font-size:1rem;color:#4a5568;font-weight:600;">Drop your CSV file here</div>
        </div>""", unsafe_allow_html=True)

# ══ TAB 3 ══
with tab3:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-size:1.4rem;font-weight:800;color:#e2e8f0;margin-bottom:0.3rem;">Model Insights</div>
      <div style="font-size:0.875rem;color:#64748b;">Complete journey from Classical ML to Deep Learning</div>
    </div>""", unsafe_allow_html=True)

    data = {
        "Model":         ["TF-IDF + LR","TF-IDF + Stacking","BiLSTM (Random)",
                          "GloVe + BiLSTM (Frozen)","GloVe + BiGRU",
                          "GloVe + BiLSTM (Fine-tuned)","GloVe + Stacked BiLSTM ⭐"],
        "Test Accuracy": [0.788,0.805,0.798,0.865,0.878,0.887,0.889],
        "Test F1":       [0.770,0.794,0.793,0.853,0.872,0.880,0.883],
        "Test AUC":      [0.874,0.889,0.887,0.944,0.950,0.955,0.957],
        "Category":      ["Classical ML","Classical ML","Deep Learning",
                          "Deep Learning","Deep Learning","Deep Learning","Deep Learning"],
    }
    df_comp = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    fig.patch.set_facecolor("#0a0a0f")
    metrics  = ["Test Accuracy","Test F1","Test AUC"]
    palettes = [["#4a5568","#6366f1","#4a5568","#8b5cf6","#a78bfa","#c4b5fd","#34d399"]]*3

    for ax, metric, pal in zip(axes, metrics, palettes):
        ax.set_facecolor("#0d0d1a")
        bars = ax.barh(df_comp["Model"], df_comp[metric], color=pal, height=0.6, edgecolor="none")
        ax.set_xlim(0.7, 1.0)
        ax.set_title(metric, color="#e2e8f0", fontweight="800", fontsize=11, pad=10)
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xlabel("Score", color="#4a5568", fontsize=9)
        ax.axvline(0.805, color="#6366f1", linestyle="--", linewidth=1, alpha=0.6)
        for bar, val in zip(bars, df_comp[metric]):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                    f"{val:.1%}", va="center", ha="left", color="#e2e8f0", fontsize=8, fontweight="700")
        ax.grid(axis="x", color="#1e1e2e", linewidth=0.8)

    fig.suptitle("Sarcasm Detection — Model Performance Journey",
                 color="#e2e8f0", fontsize=13, fontweight="900", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    styled = df_comp.copy()
    for col in ["Test Accuracy","Test F1","Test AUC"]:
        styled[col] = styled[col].map("{:.1%}".format)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1.5rem;">
      <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);
                  border-radius:14px;padding:1.2rem;">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">🧠</div>
        <div style="font-weight:800;color:#818cf8;margin-bottom:0.3rem;">Classical ML</div>
        <div style="font-size:0.8rem;color:#64748b;line-height:1.6;">
          TF-IDF (20K features, 1–3 grams) + Stacking Ensemble (LR+SVM+NB → LR). Fast, interpretable.
        </div>
        <div style="margin-top:0.8rem;font-size:1rem;font-weight:800;color:#818cf8;">80.5%</div>
      </div>
      <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
                  border-radius:14px;padding:1.2rem;">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">🔬</div>
        <div style="font-weight:800;color:#34d399;margin-bottom:0.3rem;">GloVe + Stacked BiLSTM</div>
        <div style="font-size:0.8rem;color:#64748b;line-height:1.6;">
          GloVe 100d (6B tokens) + 3-layer Bidirectional LSTM. Understands word meaning and context.
        </div>
        <div style="margin-top:0.8rem;font-size:1rem;font-weight:800;color:#34d399;">88.9% ⭐</div>
      </div>
      <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
                  border-radius:14px;padding:1.2rem;">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">⚡</div>
        <div style="font-weight:800;color:#fbbf24;margin-bottom:0.3rem;">Transformers (Next Step)</div>
        <div style="font-size:0.8rem;color:#64748b;line-height:1.6;">
          DistilBERT / RoBERTa fine-tuning. Self-attention reads full headline simultaneously.
        </div>
        <div style="margin-top:0.8rem;font-size:1rem;font-weight:800;color:#fbbf24;">90%+ 🚀</div>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:3rem;border-top:1px solid rgba(139,92,246,0.15);
            padding:1.5rem 0 0.5rem 0;text-align:center;">
  <span style="font-size:0.8rem;color:#374151;">
    SarcasmIQ · Capstone Project 2 · NLP Pipeline: Classical ML → Deep Learning → Transformers
  </span>
</div>""", unsafe_allow_html=True)
