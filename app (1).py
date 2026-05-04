import streamlit as st
import torch
import re
import html
import unicodedata
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="SuiSense",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;600;800&family=Share+Tech+Mono&display=swap');

* { font-family: 'Exo 2', sans-serif !important; }

.stApp {
    background: #050a14;
    color: #c8d8f0;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: gridMove 20s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes gridMove {
    0%   { background-position: 0 0; }
    100% { background-position: 40px 40px; }
}

/* Header */
.main-title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    background: linear-gradient(135deg, #00f5ff 0%, #0080ff 50%, #00ff88 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    animation: titlePulse 4s ease-in-out infinite;
}

@keyframes titlePulse {
    0%, 100% { filter: drop-shadow(0 0 20px rgba(0,245,255,0.3)); }
    50%       { filter: drop-shadow(0 0 50px rgba(0,245,255,0.6)); }
}

.sub-title {
    text-align: center;
    font-family: 'Share Tech Mono', monospace !important;
    color: #4a6080;
    font-size: 0.8rem;
    letter-spacing: 0.2em;
    margin-top: 4px;
    margin-bottom: 32px;
}

/* Textarea */
.stTextArea textarea {
    background: #070e1c !important;
    border: 1px solid #1a2d4a !important;
    border-radius: 12px !important;
    color: #c8d8f0 !important;
    font-size: 1rem !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}

.stTextArea textarea:focus {
    border-color: #00f5ff !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.15) !important;
}

/* Button */
.stButton button {
    width: 100%;
    background: linear-gradient(135deg, #003d5c, #005580) !important;
    border: 1px solid #00f5ff !important;
    border-radius: 12px !important;
    color: #00f5ff !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.1em !important;
    padding: 14px !important;
    transition: all 0.3s !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, #005580, #007aad) !important;
    box-shadow: 0 0 40px rgba(0,245,255,0.3) !important;
    transform: translateY(-2px) !important;
}

/* Result boxes */
.result-suicide {
    background: linear-gradient(135deg, #1a0010, #2a0018);
    border: 1px solid #ff2d55;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 40px rgba(255,45,85,0.25);
    animation: pulseDanger 1.5s ease-in-out infinite;
}

.result-normal {
    background: linear-gradient(135deg, #001a10, #002818);
    border: 1px solid #00ff88;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,255,136,0.2);
    animation: pulseSafe 2.5s ease-in-out infinite;
}

.result-uncertain {
    background: linear-gradient(135deg, #1a1500, #262000);
    border: 1px solid #ffd60a;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 40px rgba(255,214,10,0.15);
}

@keyframes pulseDanger {
    0%, 100% { box-shadow: 0 0 40px rgba(255,45,85,0.2); }
    50%       { box-shadow: 0 0 70px rgba(255,45,85,0.45); }
}

@keyframes pulseSafe {
    0%, 100% { box-shadow: 0 0 30px rgba(0,255,136,0.15); }
    50%       { box-shadow: 0 0 60px rgba(0,255,136,0.35); }
}

.result-label {
    font-size: 2.2rem;
    font-weight: 800;
    color: #fff;
    margin-bottom: 8px;
}

.result-conf {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #4a6080;
    letter-spacing: 0.1em;
}

.crisis-note {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #ff2d55;
    margin-top: 12px;
    letter-spacing: 0.08em;
}

/* Confidence bar */
.conf-bar-wrap {
    width: 100%;
    height: 4px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    margin: 12px auto;
    max-width: 200px;
    overflow: hidden;
}

.conf-bar-danger { background: linear-gradient(90deg, #8b001a, #ff2d55); height: 4px; border-radius: 4px; }
.conf-bar-safe   { background: linear-gradient(90deg, #004422, #00ff88); height: 4px; border-radius: 4px; }
.conf-bar-warn   { background: linear-gradient(90deg, #664400, #ffd60a); height: 4px; border-radius: 4px; }

/* Footer */
.footer {
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #2a3a50;
    letter-spacing: 0.12em;
    margin-top: 40px;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Config ───────────────────────────────────────────────────
MODEL_REPO       = "Shahid-0039/distilbert-suicide-detection"
MODEL_SUBFOLDER  = "roberta_model"
MAX_LEN          = 256
MIN_WORDS        = 10
CONFIDENCE_THRESHOLD = 0.60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(
        MODEL_REPO, subfolder=MODEL_SUBFOLDER)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_REPO, subfolder=MODEL_SUBFOLDER)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# ── Text Cleaning ────────────────────────────────────────────
def fix_encoding(text):
    try:
        text = text.encode('latin-1').decode('utf-8')
    except Exception:
        pass
    text = unicodedata.normalize('NFKD', text)
    for bad, good in {
        'â€™':"'", 'â€œ':'"', 'â€':'"',
        'Â':'', '&amp;':'&', '&lt;':'<', '&gt;':'>'
    }.items():
        text = text.replace(bad, good)
    return text

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = fix_encoding(text)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Predict ──────────────────────────────────────────────────
def predict(text, tokenizer, model):
    cleaned = clean_text(text)
    enc = tokenizer(cleaned, max_length=MAX_LEN, padding='max_length',
                    truncation=True, return_tensors='pt')
    with torch.no_grad():
        out   = model(input_ids=enc['input_ids'].to(DEVICE),
                      attention_mask=enc['attention_mask'].to(DEVICE))
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]

    pred       = int(probs.argmax())
    confidence = float(probs[pred])
    return pred, confidence, probs

# ── UI ───────────────────────────────────────────────────────
st.markdown("<div class='main-title'>⬡ SUISENSE</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>SUICIDE RISK DETECTION · ROBERTA · NLP</div>",
            unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    tokenizer, model = load_model()

# Input
text_input = st.text_area(
    "INPUT TEXT  [ MINIMUM 10 WORDS ]",
    placeholder="Type or paste text here...",
    height=160,
    key="input"
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze = st.button("⬡  ANALYZE", use_container_width=True)
with col2:
    clear = st.button("CLEAR", use_container_width=True)

if clear:
    st.rerun()

# Result
if analyze:
    if not text_input or not text_input.strip():
        st.warning("⚠ Please enter some text.")
    elif len(text_input.strip().split()) < MIN_WORDS:
        st.warning(f"⚠ Minimum {MIN_WORDS} words required. You entered {len(text_input.strip().split())} words.")
    else:
        with st.spinner("Analyzing..."):
            pred, confidence, probs = predict(text_input, tokenizer, model)

        pct = round(confidence * 100, 1)

        if confidence < CONFIDENCE_THRESHOLD:
            st.markdown(f"""
            <div class='result-uncertain'>
                <div class='result-label'>🟡 Uncertain</div>
                <div class='conf-bar-wrap'><div class='conf-bar-warn' style='width:{pct}%'></div></div>
                <div class='result-conf'>CONFIDENCE: {pct}% — Human review needed</div>
            </div>""", unsafe_allow_html=True)

        elif pred == 1:
            st.markdown(f"""
            <div class='result-suicide'>
                <div class='result-label'>🔴 Suicide</div>
                <div class='conf-bar-wrap'><div class='conf-bar-danger' style='width:{pct}%'></div></div>
                <div class='result-conf'>CONFIDENCE: {pct}%</div>
                <div class='crisis-note'>⚠ Crisis helpline: Umang 0311-7786264</div>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class='result-normal'>
                <div class='result-label'>🟢 Normal</div>
                <div class='conf-bar-wrap'><div class='conf-bar-safe' style='width:{pct}%'></div></div>
                <div class='result-conf'>CONFIDENCE: {pct}%</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<div class='footer'>⚠ RESEARCH USE ONLY · NOT A CLINICAL TOOL · SuiSense v1.0</div>",
            unsafe_allow_html=True)
