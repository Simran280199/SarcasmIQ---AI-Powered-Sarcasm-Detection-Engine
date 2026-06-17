"""
app.py - Capstone Project 2: Sarcasm Detection
Ultra-Professional Streamlit UI
"""

import re, pickle, os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

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
# NLTK
# ─────────────────────────────────────────────
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try: nltk.data.find(f"corpora/{pkg}")
        except: nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

STOPWORDS, LEMMATIZER = setup_nltk()

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0a0f; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #0a0a12 100%);
    border-right: 1px solid rgba(139,92,246,0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.8rem !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(139,92,246,0.1) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Textarea ── */
.stTextArea textarea {
    background: rgba(15,15,30,0.8) !important;
    border: 2px solid rgba(139,92,246,0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.3s ease;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(139,92,246,0.8) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}
.stTextArea textarea::placeholder { color: #4a5568 !important; }

/* ── Buttons ── */
.stButton > button {
    width: 100%; border-radius: 12px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    transition: all 0.3s ease !important; border: none !important;
    padding: 0.75rem 1.5rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,15,30,0.5);
    border-radius: 12px; padding: 4px; gap: 4px;
    border: 1px solid rgba(139,92,246,0.2);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; color: #64748b !important;
    font-weight: 600 !important; padding: 0.6rem 1.2rem !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    color: white !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px; padding: 1rem;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-size:0.8rem !important; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #e2e8f0 !important; font-size:1.4rem !important; font-weight:700 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(15,15,30,0.5) !important;
    border: 2px dashed rgba(139,92,246,0.3) !important;
    border-radius: 12px !important; padding: 1rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: rgba(139,92,246,0.2) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(139,92,246,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important; font-weight: 600 !important;
}

/* ── Scrollbar ── */
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
    except: return None, None, {}, False

@st.cache_resource
def load_dl_model():
    try:
        import tensorflow as tf
        model     = tf.keras.models.load_model("models/best_dl_model.keras")
        tokenizer = pickle.load(open("models/dl_tokenizer.pkl","rb"))
        meta      = pickle.load(open("models/dl_metadata.pkl","rb"))
        return model, tokenizer, meta, True
    except: return None, None, {}, False

tfidf, ml_model, ml_meta, ML_OK = load_ml_model()
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
    p = float(dl_model.predict(padded, verbose=0)[0][0])
    return p, 1-p

# ─────────────────────────────────────────────
# ── NAVBAR / HERO HEADER ──
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d0d1a 0%, #130d2e 50%, #0d0d1a 100%);
    border-bottom: 1px solid rgba(139,92,246,0.3);
    padding: 1.5rem 2rem 1.2rem 2rem;
    margin: -2rem -2rem 2rem -2rem;
    position: relative; overflow: hidden;
">
  <!-- Glow blobs -->
  <div style="position:absolute;top:-60px;left:10%;width:300px;height:300px;
              background:radial-gradient(circle,rgba(124,58,237,0.15),transparent 70%);
              pointer-events:none;"></div>
  <div style="position:absolute;top:-40px;right:15%;width:250px;height:250px;
              background:radial-gradient(circle,rgba(79,70,229,0.12),transparent 70%);
              pointer-events:none;"></div>

  <div style="display:flex; align-items:center; justify-content:space-between; position:relative;">
    <div style="display:flex; align-items:center; gap:1rem;">
      <div style="
          background: linear-gradient(135deg,#7c3aed,#4f46e5);
          border-radius:14px; width:48px; height:48px;
          display:flex; align-items:center; justify-content:center;
          font-size:1.6rem; box-shadow:0 4px 20px rgba(124,58,237,0.5);">
        🎭
      </div>
      <div>
        <div style="font-size:1.6rem; font-weight:900; color:white; letter-spacing:-0.5px;
                    background:linear-gradient(90deg,#a78bfa,#818cf8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
          SarcasmIQ
        </div>
        <div style="font-size:0.75rem; color:#64748b; font-weight:500; letter-spacing:1px; text-transform:uppercase;">
          AI-Powered Sarcasm Detection Engine
        </div>
      </div>
    </div>
    <div style="display:flex; gap:0.6rem; flex-wrap:wrap;">
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
    <div style="padding:0.5rem 0 1rem 0;">
      <div style="font-size:0.7rem;color:#4a5568;letter-spacing:2px;text-transform:uppercase;
                  font-weight:700;margin-bottom:0.8rem;">⚙️ Configuration</div>
    </div>
    """, unsafe_allow_html=True)

    model_options = []
    if ML_OK: model_options.append("🧠 Classical ML  (80.5%)")
    if DL_OK: model_options.append("🔬 GloVe + Stacked BiLSTM  (88.9%)")
    if not model_options:
        st.error("No models found! Run train_v2.py first.")
        st.stop()

    selected = st.selectbox("Active Model", model_options)
    threshold = st.slider("Sarcasm Threshold", 0.0, 1.0, 0.5, 0.01,
                          help="Probability above this = Sarcastic")

    st.markdown("<hr style='margin:1.2rem 0'>", unsafe_allow_html=True)

    # Performance card
    if "Classical" in selected and ML_OK:
        acc = ml_meta.get("test_accuracy", 0)
        f1  = ml_meta.get("test_f1", 0)
        auc = ml_meta.get("test_auc", 0)
        mname = "Stacking Ensemble"
    else:
        acc = dl_meta.get("test_accuracy", 0)
        f1  = dl_meta.get("test_f1", 0)
        auc = dl_meta.get("test_auc", 0)
        mname = "GloVe + Stacked BiLSTM"

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
        <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:0.6rem;text-align:center;grid-column:span 2;">
          <div style="font-size:1.1rem;font-weight:800;color:#fbbf24;">{auc:.1%}</div>
          <div style="font-size:0.7rem;color:#64748b;">ROC-AUC</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.2rem 0'>", unsafe_allow_html=True)

    # Journey
    st.markdown("""
    <div style="font-size:0.7rem;color:#4a5568;letter-spacing:2px;text-transform:uppercase;
                font-weight:700;margin-bottom:0.8rem;">🚀 Accuracy Journey</div>
    """, unsafe_allow_html=True)

    journey = [
        ("TF-IDF + LR",         0.788, "#4a5568"),
        ("TF-IDF + Ensemble",   0.805, "#6366f1"),
        ("BiLSTM (random)",     0.798, "#4a5568"),
        ("GloVe + BiLSTM",      0.865, "#8b5cf6"),
        ("GloVe + BiGRU",       0.877, "#a78bfa"),
        ("GloVe + Stacked ✅",  0.889, "#34d399"),
    ]
    for name, val, color in journey:
        pct = int(val * 100)
        bar_w = int(val * 130)
        st.markdown(f"""
        <div style="margin-bottom:0.5rem;">
          <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
            <span style="font-size:0.72rem;color:#94a3b8;">{name}</span>
            <span style="font-size:0.72rem;color:{color};font-weight:700;">{pct}%</span>
          </div>
          <div style="background:rgba(255,255,255,0.05);border-radius:999px;height:5px;">
            <div style="width:{bar_w}px;max-width:100%;background:{color};
                        border-radius:999px;height:5px;transition:width 0.5s;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem;color:#4a5568;text-align:center;line-height:1.6;">
      Built with ❤️ | Streamlit + TensorFlow<br>
      Capstone Project 2 — NLP
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍  Single Prediction",
    "📁  Batch Analysis",
    "📊  Model Insights",
])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════
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
            "headline_input", height=130, label_visibility="collapsed",
            placeholder="e.g.  'Scientists discover water is, in fact, wet' ...",
        )

        c1, c2 = st.columns([2, 1])
        predict_btn = c1.button("🔮  Detect Sarcasm", type="primary", use_container_width=True)
        clear_btn   = c2.button("✕  Clear", type="secondary", use_container_width=True)

        # Sample headlines
        st.markdown("""
        <div style="margin-top:1rem;">
          <div style="font-size:0.75rem;color:#4a5568;font-weight:600;
                      text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;">
            Try a sample
          </div>
        </div>
        """, unsafe_allow_html=True)

        samples = [
            "🎭  Scientists discover water is, in fact, wet",
            "🎭  Area man passionate defender of what he calls 'my team'",
            "📰  Stock market hits record high amid strong earnings",
            "📰  Local firefighters rescue family from burning building",
        ]
        for s in samples:
            if st.button(s, key=s, use_container_width=True):
                user_input = s.split("  ", 1)[1]
                predict_btn = True

    with col_right:
        if predict_btn and user_input and user_input.strip():
            with st.spinner(""):
                try:
                    if "Classical" in selected and ML_OK:
                        sarc_p, not_p = predict_ml(user_input)
                    else:
                        sarc_p, not_p = predict_dl(user_input)

                    is_sarc = sarc_p >= threshold
                    conf    = sarc_p if is_sarc else not_p

                    # ── Result Hero Card ──
                    if is_sarc:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg,#7c1d1d,#991b1b,#7c1d1d);
                            border: 1px solid rgba(239,68,68,0.5);
                            border-radius: 20px; padding: 2rem; text-align: center;
                            box-shadow: 0 0 40px rgba(239,68,68,0.2);
                            position: relative; overflow: hidden;
                        ">
                          <div style="font-size:3.5rem;margin-bottom:0.5rem;">😏</div>
                          <div style="font-size:1.8rem;font-weight:900;color:#fca5a5;
                                      letter-spacing:-0.5px;margin-bottom:0.3rem;">
                            Sarcastic!
                          </div>
                          <div style="font-size:0.85rem;color:#f87171;margin-bottom:1.2rem;">
                            This text carries sarcastic intent
                          </div>
                          <div style="background:rgba(0,0,0,0.3);border-radius:12px;
                                      padding:0.8rem;display:inline-block;min-width:160px;">
                            <div style="font-size:2rem;font-weight:900;color:white;">
                              {conf:.1%}
                            </div>
                            <div style="font-size:0.75rem;color:#fca5a5;font-weight:600;">CONFIDENCE</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg,#064e3b,#065f46,#064e3b);
                            border: 1px solid rgba(16,185,129,0.5);
                            border-radius: 20px; padding: 2rem; text-align: center;
                            box-shadow: 0 0 40px rgba(16,185,129,0.2);
                        ">
                          <div style="font-size:3.5rem;margin-bottom:0.5rem;">🙂</div>
                          <div style="font-size:1.8rem;font-weight:900;color:#6ee7b7;
                                      letter-spacing:-0.5px;margin-bottom:0.3rem;">
                            Not Sarcastic
                          </div>
                          <div style="font-size:0.85rem;color:#34d399;margin-bottom:1.2rem;">
                            This text appears genuine
                          </div>
                          <div style="background:rgba(0,0,0,0.3);border-radius:12px;
                                      padding:0.8rem;display:inline-block;min-width:160px;">
                            <div style="font-size:2rem;font-weight:900;color:white;">
                              {conf:.1%}
                            </div>
                            <div style="font-size:0.75rem;color:#6ee7b7;font-weight:600;">CONFIDENCE</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("")

                    # ── Probability Breakdown ──
                    st.markdown("""
                    <div style="font-size:0.75rem;color:#4a5568;font-weight:700;
                                text-transform:uppercase;letter-spacing:1px;margin-bottom:0.6rem;">
                      Probability Breakdown
                    </div>
                    """, unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    m1.metric("🙂 Not Sarcastic", f"{not_p:.1%}")
                    m2.metric("😏 Sarcastic", f"{sarc_p:.1%}")

                    # ── Custom probability bar ──
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
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Cross-model comparison ──
                    if ML_OK and DL_OK:
                        with st.expander("🔁 Compare across models"):
                            rows = []
                            p_ml, _ = predict_ml(user_input)
                            p_dl, _ = predict_dl(user_input)
                            for mname, prob in [("Classical ML", p_ml), ("GloVe + Stacked BiLSTM", p_dl)]:
                                rows.append({
                                    "Model": mname,
                                    "Sarcasm %": f"{prob:.1%}",
                                    "Verdict": "😏 Sarcastic" if prob >= threshold else "🙂 Not Sarcastic",
                                    "Confidence": f"{max(prob, 1-prob):.1%}"
                                })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

        elif not user_input or not user_input.strip():
            # Placeholder when empty
            st.markdown("""
            <div style="
                border: 2px dashed rgba(139,92,246,0.2);
                border-radius: 20px; padding: 3rem; text-align: center;
                background: rgba(139,92,246,0.03);
            ">
              <div style="font-size:3rem;margin-bottom:1rem;opacity:0.4;">🎭</div>
              <div style="font-size:1rem;color:#4a5568;font-weight:600;">
                Your result will appear here
              </div>
              <div style="font-size:0.8rem;color:#374151;margin-top:0.3rem;">
                Enter a headline and click Detect Sarcasm
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — BATCH
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-size:1.4rem;font-weight:800;color:#e2e8f0;margin-bottom:0.3rem;">
        Batch Analysis
      </div>
      <div style="font-size:0.875rem;color:#64748b;">
        Upload a CSV file to analyze hundreds of headlines at once
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
                    border-radius:10px;padding:0.8rem 1rem;margin-bottom:1rem;
                    display:flex;align-items:center;gap:0.5rem;">
          <span style="color:#34d399;font-weight:700;">✓</span>
          <span style="color:#6ee7b7;font-size:0.875rem;">
            Loaded <b>{len(df_batch):,}</b> rows · <b>{len(df_batch.columns)}</b> columns
          </span>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(df_batch.head(), use_container_width=True, hide_index=True)
        col_choice = st.selectbox("Select text column", df_batch.columns)

        if st.button("⚡ Run Batch Analysis", type="primary"):
            prog_bar = st.progress(0, text="Analyzing...")
            probs = []
            for i, text in enumerate(df_batch[col_choice].astype(str)):
                try:
                    if "Classical" in selected and ML_OK:
                        p, _ = predict_ml(text)
                    else:
                        p, _ = predict_dl(text)
                    probs.append(p)
                except: probs.append(0.5)
                if i % 10 == 0:
                    prog_bar.progress((i+1)/len(df_batch),
                                      text=f"Processing {i+1}/{len(df_batch)}...")

            prog_bar.empty()
            df_batch["sarcasm_probability"] = probs
            df_batch["prediction"] = ["😏 Sarcastic" if p>=threshold
                                      else "🙂 Not Sarcastic" for p in probs]
            df_batch["confidence"] = [f"{max(p,1-p):.1%}" for p in probs]

            n_sarc     = sum(1 for p in probs if p >= threshold)
            n_not_sarc = len(probs) - n_sarc

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Headlines", f"{len(df_batch):,}")
            m2.metric("😏 Sarcastic",    f"{n_sarc:,}", f"{n_sarc/len(probs):.1%}")
            m3.metric("🙂 Not Sarcastic", f"{n_not_sarc:,}", f"{n_not_sarc/len(probs):.1%}")
            m4.metric("Avg Confidence",  f"{np.mean([max(p,1-p) for p in probs]):.1%}")

            st.dataframe(df_batch, use_container_width=True, hide_index=True)

            csv_out = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV", csv_out,
                "sarcasm_predictions.csv", "text/csv",
                use_container_width=True
            )
    else:
        st.markdown("""
        <div style="border:2px dashed rgba(139,92,246,0.2);border-radius:20px;
                    padding:4rem;text-align:center;">
          <div style="font-size:3rem;margin-bottom:1rem;opacity:0.3;">📁</div>
          <div style="font-size:1rem;color:#4a5568;font-weight:600;">Drop your CSV file here</div>
          <div style="font-size:0.8rem;color:#374151;margin-top:0.3rem;">
            Must contain a column with text/headlines
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-size:1.4rem;font-weight:800;color:#e2e8f0;margin-bottom:0.3rem;">
        Model Insights & Journey
      </div>
      <div style="font-size:0.875rem;color:#64748b;">
        Complete evolution from Classical ML to Deep Learning
      </div>
    </div>
    """, unsafe_allow_html=True)

    data = {
        "Model": [
            "TF-IDF + Logistic Regression",
            "TF-IDF + Stacking Ensemble",
            "BiLSTM (Random Embeddings)",
            "GloVe + BiLSTM (Frozen)",
            "GloVe + BiGRU",
            "GloVe + BiLSTM (Fine-tuned)",
            "GloVe + Stacked BiLSTM ⭐",
        ],
        "Test Accuracy": [0.788,0.805,0.798,0.865,0.878,0.887,0.889],
        "Test F1":       [0.770,0.794,0.793,0.853,0.872,0.880,0.883],
        "Test AUC":      [0.874,0.889,0.887,0.944,0.950,0.955,0.957],
        "Category":      ["Classical ML","Classical ML","Deep Learning",
                          "Deep Learning","Deep Learning","Deep Learning","Deep Learning"],
    }
    df_comp = pd.DataFrame(data)

    # ── Matplotlib dark chart ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0a0a0f")

    metrics  = ["Test Accuracy","Test F1","Test AUC"]
    palettes = [
        ["#4a5568","#6366f1","#4a5568","#8b5cf6","#a78bfa","#c4b5fd","#34d399"],
        ["#4a5568","#6366f1","#4a5568","#8b5cf6","#a78bfa","#c4b5fd","#34d399"],
        ["#4a5568","#6366f1","#4a5568","#8b5cf6","#a78bfa","#c4b5fd","#34d399"],
    ]

    for ax, metric, pal in zip(axes, metrics, palettes):
        ax.set_facecolor("#0d0d1a")
        bars = ax.barh(df_comp["Model"], df_comp[metric],
                       color=pal, height=0.6, edgecolor="none")
        ax.set_xlim(0.7, 1.0)
        ax.set_title(metric, color="#e2e8f0", fontweight="800",
                     fontsize=11, pad=10, fontfamily="sans-serif")
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.xaxis.set_tick_params(colors="#374151")
        ax.yaxis.set_tick_params(colors="#94a3b8")
        ax.set_xlabel("Score", color="#4a5568", fontsize=9)
        ax.axvline(0.805, color="#6366f1", linestyle="--",
                   linewidth=1, alpha=0.6, label="ML Baseline")
        for bar, val in zip(bars, df_comp[metric]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.1%}", va="center", ha="left",
                    color="#e2e8f0", fontsize=8, fontweight="700")
        ax.grid(axis="x", color="#1e1e2e", linewidth=0.8)

    fig.suptitle("Sarcasm Detection — Model Performance Journey",
                 color="#e2e8f0", fontsize=13, fontweight="900", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.write("")

    # ── Styled table ──
    st.markdown("""
    <div style="font-size:0.75rem;color:#4a5568;font-weight:700;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;">
      Detailed Comparison
    </div>
    """, unsafe_allow_html=True)

    styled_df = df_comp.copy()
    styled_df["Test Accuracy"] = styled_df["Test Accuracy"].map("{:.1%}".format)
    styled_df["Test F1"]       = styled_df["Test F1"].map("{:.1%}".format)
    styled_df["Test AUC"]      = styled_df["Test AUC"].map("{:.1%}".format)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ── Architecture cards ──
    st.write("")
    st.markdown("""
    <div style="font-size:0.75rem;color:#4a5568;font-weight:700;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:1rem;">
      Architecture Highlights
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
      <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);
                  border-radius:14px;padding:1.2rem;">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">🧠</div>
        <div style="font-weight:800;color:#818cf8;margin-bottom:0.3rem;">Classical ML</div>
        <div style="font-size:0.8rem;color:#64748b;line-height:1.6;">
          TF-IDF (20K features, 1–3 grams) + Stacking Ensemble
          (LR + SVM + NB → LR meta-learner). Fast, interpretable.
        </div>
        <div style="margin-top:0.8rem;font-size:1rem;font-weight:800;color:#818cf8;">80.5%</div>
      </div>
      <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
                  border-radius:14px;padding:1.2rem;">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">🔬</div>
        <div style="font-weight:800;color:#34d399;margin-bottom:0.3rem;">GloVe + Stacked BiLSTM</div>
        <div style="font-size:0.8rem;color:#64748b;line-height:1.6;">
          GloVe 100d (6B tokens) + 3-layer Bidirectional LSTM.
          Understands word meaning and sequence context.
        </div>
        <div style="margin-top:0.8rem;font-size:1rem;font-weight:800;color:#34d399;">88.9% ⭐</div>
      </div>
      <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
                  border-radius:14px;padding:1.2rem;">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">⚡</div>
        <div style="font-weight:800;color:#fbbf24;margin-bottom:0.3rem;">Transformers (Next Step)</div>
        <div style="font-size:0.8rem;color:#64748b;line-height:1.6;">
          DistilBERT / RoBERTa fine-tuning. Self-attention reads
          the full headline simultaneously — best for sarcasm.
        </div>
        <div style="margin-top:0.8rem;font-size:1rem;font-weight:800;color:#fbbf24;">90%+ 🚀</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style="
    margin-top:3rem;
    border-top:1px solid rgba(139,92,246,0.15);
    padding:1.5rem 0 0.5rem 0;
    text-align:center;
">
  <span style="font-size:0.8rem;color:#374151;">
    SarcasmIQ · Capstone Project 2 · NLP Pipeline: Classical ML → Deep Learning → Transformers
    · Built with Streamlit & TensorFlow
  </span>
</div>
""", unsafe_allow_html=True)
