import streamlit as st
import torch
import re
import html
import unicodedata
from transformers import RobertaTokenizer, RobertaForSequenceClassification

st.set_page_config(
    page_title="SuiSense",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&family=Share+Tech+Mono&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #000408 !important;
    color: #e0f0ff !important;
    font-family: 'Rajdhani', sans-serif !important;
    overflow-x: hidden;
}

.stApp::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(0,180,255,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(120,0,255,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(0,255,150,0.05) 0%, transparent 60%);
    animation: auroraShift 12s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes auroraShift {
    0%   { transform: translate(0,0) rotate(0deg); }
    100% { transform: translate(-3%,2%) rotate(1deg); }
}

.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='52'%3E%3Cpolygon points='30,2 58,17 58,37 30,52 2,37 2,17' fill='none' stroke='rgba(0,180,255,0.04)' stroke-width='1'/%3E%3C/svg%3E");
    background-size: 60px 52px;
    animation: hexDrift 30s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes hexDrift {
    0%   { background-position: 0 0; }
    100% { background-position: 60px 104px; }
}

.block-container {
    position: relative;
    z-index: 1;
    max-width: 800px !important;
    padding: 2rem 2rem 4rem !important;
}

.sui-header { text-align: center; padding: 20px 0 10px; }

.sui-logo-ring {
    width: 100px; height: 100px;
    margin: 0 auto 16px;
    position: relative;
}

.sui-logo-ring svg {
    width: 100%; height: 100%;
    animation: ringRotate 8s linear infinite;
    filter: drop-shadow(0 0 12px rgba(0,200,255,0.6));
}

@keyframes ringRotate { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

.sui-logo-inner {
    position: absolute;
    inset: 20px;
    background: radial-gradient(circle, rgba(0,200,255,0.15), transparent);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    animation: innerPulse 3s ease-in-out infinite;
}

@keyframes innerPulse {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.1); }
}

.sui-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2.8rem, 8vw, 5rem);
    font-weight: 900;
    letter-spacing: 0.3em;
    background: linear-gradient(135deg, #00c8ff 0%, #ffffff 30%, #00ffaa 60%, #b060ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 300% 300%;
    animation: titleGradient 6s ease infinite;
    margin-bottom: 8px;
}

@keyframes titleGradient {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.sui-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.35em;
    color: rgba(100,180,255,0.5);
    margin-bottom: 8px;
}

.scan-line {
    width: 80%; max-width: 500px;
    height: 1px; margin: 16px auto;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.3), rgba(0,255,150,0.6), rgba(0,200,255,0.3), transparent);
    position: relative; overflow: hidden;
}

.scan-line::after {
    content: '';
    position: absolute;
    top: -2px; left: -30%;
    width: 30%; height: 5px;
    background: linear-gradient(90deg, transparent, #00ffaa, transparent);
    animation: scanMove 3s ease-in-out infinite;
    filter: blur(2px);
}

@keyframes scanMove { 0% { left: -30%; } 100% { left: 130%; } }

.stTextArea > label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.2em !important;
    color: rgba(0,200,255,0.5) !important;
}

.stTextArea textarea {
    background: rgba(0,8,20,0.8) !important;
    border: 1px solid rgba(0,150,255,0.2) !important;
    border-radius: 12px !important;
    color: #c8e8ff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    transition: all 0.4s ease !important;
    caret-color: #00ffaa !important;
}

.stTextArea textarea:focus {
    border-color: rgba(0,200,255,0.5) !important;
    box-shadow: 0 0 30px rgba(0,200,255,0.08) !important;
}

.stButton > button {
    width: 100% !important;
    border-radius: 12px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.15em !important;
    padding: 16px 24px !important;
    transition: all 0.3s ease !important;
}

div[data-testid="column"]:first-child .stButton > button {
    background: linear-gradient(135deg, #003a5c, #005a8a, #003a5c) !important;
    border: 1px solid rgba(0,200,255,0.4) !important;
    color: #00c8ff !important;
    box-shadow: 0 0 20px rgba(0,180,255,0.1) !important;
}

div[data-testid="column"]:first-child .stButton > button:hover {
    box-shadow: 0 0 50px rgba(0,200,255,0.3) !important;
    transform: translateY(-3px) !important;
}

div[data-testid="column"]:last-child .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(80,100,140,0.3) !important;
    color: rgba(100,140,200,0.5) !important;
}

.result-box {
    border-radius: 20px;
    padding: 36px 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 24px;
}

.orb-3d {
    width: 90px; height: 90px;
    border-radius: 50%;
    margin: 0 auto 20px;
    position: relative;
    animation: orbFloat 4s ease-in-out infinite;
}

@keyframes orbFloat {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-14px); }
}

.orb-3d::before {
    content: '';
    position: absolute;
    inset: -10px;
    border-radius: 50%;
    border: 2px solid transparent;
    animation: orbRing1 3s linear infinite;
}

.orb-3d::after {
    content: '';
    position: absolute;
    inset: -20px;
    border-radius: 50%;
    border: 1px solid transparent;
    animation: orbRing2 5s linear infinite reverse;
}

@keyframes orbRing1 { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
@keyframes orbRing2 { 0% { transform: rotate(0deg); } 100% { transform: rotate(-360deg); } }

.result-danger {
    background: linear-gradient(135deg, rgba(20,0,8,0.98), rgba(35,0,15,0.95));
    border: 1px solid rgba(255,30,70,0.3);
    animation: dangerPulse 2s ease-in-out infinite;
}

@keyframes dangerPulse {
    0%, 100% { box-shadow: 0 0 60px rgba(255,30,70,0.15); }
    50%       { box-shadow: 0 0 100px rgba(255,30,70,0.35); }
}

.orb-danger {
    background: radial-gradient(circle at 35% 30%, #ff8099, #ff1e46, #a00020, #400010);
    box-shadow: 0 0 50px rgba(255,30,70,0.7), 0 0 100px rgba(255,30,70,0.3);
}
.orb-danger::before { border-top-color: rgba(255,30,70,0.9); border-right-color: rgba(255,30,70,0.4); }
.orb-danger::after  { border-bottom-color: rgba(255,100,120,0.3); }

.result-safe {
    background: linear-gradient(135deg, rgba(0,15,8,0.98), rgba(0,25,15,0.95));
    border: 1px solid rgba(0,255,130,0.25);
    animation: safePulse 3s ease-in-out infinite;
}

@keyframes safePulse {
    0%, 100% { box-shadow: 0 0 60px rgba(0,255,130,0.1); }
    50%       { box-shadow: 0 0 100px rgba(0,255,130,0.25); }
}

.orb-safe {
    background: radial-gradient(circle at 35% 30%, #80ffcc, #00ff88, #00a050, #003020);
    box-shadow: 0 0 50px rgba(0,255,130,0.7), 0 0 100px rgba(0,255,130,0.3);
}
.orb-safe::before { border-top-color: rgba(0,255,130,0.9); border-right-color: rgba(0,255,130,0.4); }
.orb-safe::after  { border-bottom-color: rgba(80,255,180,0.3); }

.result-uncertain {
    background: linear-gradient(135deg, rgba(15,12,0,0.98), rgba(25,20,0,0.95));
    border: 1px solid rgba(255,200,0,0.25);
    box-shadow: 0 0 60px rgba(255,200,0,0.1);
}

.orb-uncertain {
    background: radial-gradient(circle at 35% 30%, #ffe880, #ffcc00, #aa8000, #443200);
    box-shadow: 0 0 50px rgba(255,200,0,0.7), 0 0 100px rgba(255,200,0,0.3);
}
.orb-uncertain::before { border-top-color: rgba(255,200,0,0.9); }

.result-label-text {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: 0.1em;
    color: #fff;
    margin-bottom: 12px;
}

.result-conf-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    opacity: 0.5;
    margin-bottom: 16px;
}

.conf-track {
    width: 60%; height: 3px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    margin: 0 auto 16px;
    overflow: hidden;
}

.conf-fill { height: 100%; border-radius: 3px; transition: width 1s ease; }
.fill-danger    { background: linear-gradient(90deg, #600010, #ff1e46); box-shadow: 0 0 10px #ff1e46; }
.fill-safe      { background: linear-gradient(90deg, #004428, #00ff88); box-shadow: 0 0 10px #00ff88; }
.fill-uncertain { background: linear-gradient(90deg, #664400, #ffcc00); box-shadow: 0 0 10px #ffcc00; }

.crisis-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    animation: crisisBlink 2s ease-in-out infinite;
}

@keyframes crisisBlink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

.sui-footer {
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: rgba(50,80,120,0.5);
    letter-spacing: 0.18em;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(0,100,200,0.08);
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Config ───────────────────────────────────────────────────
MODEL_REPO       = "Shahid-0039/distilbert-suicide-detection"
MODEL_SUBFOLDER  = "roberta_model"
MAX_LEN          = 256
MIN_WORDS        = 10
CONFIDENCE_THRESHOLD = 0.60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    tok = RobertaTokenizer.from_pretrained(MODEL_REPO, subfolder=MODEL_SUBFOLDER)
    mdl = RobertaForSequenceClassification.from_pretrained(MODEL_REPO, subfolder=MODEL_SUBFOLDER)
    mdl.to(DEVICE); mdl.eval()
    return tok, mdl

def clean_text(text):
    if not isinstance(text, str) or not text.strip(): return ""
    try: text = text.encode('latin-1').decode('utf-8')
    except: pass
    text = unicodedata.normalize('NFKD', text)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict(text, tok, mdl):
    enc = tok(clean_text(text), max_length=MAX_LEN, padding='max_length',
               truncation=True, return_tensors='pt')
    with torch.no_grad():
        out   = mdl(input_ids=enc['input_ids'].to(DEVICE),
                    attention_mask=enc['attention_mask'].to(DEVICE))
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
    return int(probs.argmax()), float(probs.max())

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class='sui-header'>
  <div class='sui-logo-ring'>
    <svg viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'>
      <polygon points='50,4 96,27 96,73 50,96 4,73 4,27'
               fill='none' stroke='url(#hg)' stroke-width='1.5' stroke-dasharray='8 4'/>
      <polygon points='50,15 85,32 85,68 50,85 15,68 15,32'
               fill='none' stroke='rgba(0,255,150,0.25)' stroke-width='1'/>
      <defs>
        <linearGradient id='hg' x1='0%' y1='0%' x2='100%' y2='100%'>
          <stop offset='0%' stop-color='#00c8ff' stop-opacity='0.9'/>
          <stop offset='100%' stop-color='#00ff88' stop-opacity='0.9'/>
        </linearGradient>
      </defs>
    </svg>
    <div class='sui-logo-inner'>🧠</div>
  </div>
  <div class='sui-title'>SUISENSE</div>
  <div class='sui-subtitle'>SUICIDE RISK DETECTION · ROBERTA · NLP · v1.0</div>
  <div class='scan-line'></div>
</div>
""", unsafe_allow_html=True)

with st.spinner("🔄 Initializing neural network..."):
    tokenizer, model = load_model()

text_input = st.text_area(
    "INPUT TEXT  ·  MINIMUM 10 WORDS",
    placeholder="Enter text for suicide risk analysis...",
    height=180, key="text_in"
)

wc = len(text_input.strip().split()) if text_input.strip() else 0
if text_input.strip():
    col = "#00ff88" if wc >= MIN_WORDS else "#ff6040"
    st.markdown(
        f"<p style='font-family:Share Tech Mono;font-size:0.68rem;color:{col};"
        f"letter-spacing:0.15em;text-align:right;margin-top:-12px;'>{wc} WORDS</p>",
        unsafe_allow_html=True)

c1, c2 = st.columns([3, 1])
with c1: analyze = st.button("⬡  ANALYZE", use_container_width=True)
with c2: clear   = st.button("CLR",        use_container_width=True)

if clear: st.rerun()

if analyze:
    if not text_input or not text_input.strip():
        st.warning("⚠  ENTER SOME TEXT TO ANALYZE")
    elif wc < MIN_WORDS:
        st.warning(f"⚠  MINIMUM {MIN_WORDS} WORDS REQUIRED  ·  YOU ENTERED {wc}")
    else:
        with st.spinner("⚡ Analyzing neural patterns..."):
            pred, conf = predict(text_input, tokenizer, model)

        pct = round(conf * 100, 1)

        if conf < CONFIDENCE_THRESHOLD:
            st.markdown(f"""
            <div class='result-box result-uncertain'>
                <div class='orb-3d orb-uncertain'></div>
                <div class='result-label-text' style='color:#ffcc00;text-shadow:0 0 30px #ffcc00;'>UNCERTAIN</div>
                <div class='conf-track'><div class='conf-fill fill-uncertain' style='width:{pct}%'></div></div>
                <div class='result-conf-text'>CONFIDENCE · {pct}%</div>
                <div class='crisis-text' style='color:rgba(255,200,0,0.7);'>⚠ LOW CONFIDENCE · HUMAN REVIEW RECOMMENDED</div>
            </div>""", unsafe_allow_html=True)

        elif pred == 1:
            st.markdown(f"""
            <div class='result-box result-danger'>
                <div class='orb-3d orb-danger'></div>
                <div class='result-label-text' style='color:#ff4060;text-shadow:0 0 40px #ff1e46;'>🔴 SUICIDE RISK</div>
                <div class='conf-track'><div class='conf-fill fill-danger' style='width:{pct}%'></div></div>
                <div class='result-conf-text'>CONFIDENCE · {pct}%</div>
                <div class='crisis-text' style='color:rgba(255,80,100,0.9);'>⚠ CRISIS HELPLINE · UMANG: 0311-7786264</div>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class='result-box result-safe'>
                <div class='orb-3d orb-safe'></div>
                <div class='result-label-text' style='color:#00ff88;text-shadow:0 0 40px #00ff88;'>🟢 NORMAL</div>
                <div class='conf-track'><div class='conf-fill fill-safe' style='width:{pct}%'></div></div>
                <div class='result-conf-text'>CONFIDENCE · {pct}%</div>
            </div>""", unsafe_allow_html=True)

st.markdown("""
<div class='sui-footer'>
    ⚠ RESEARCH USE ONLY · NOT A CLINICAL DIAGNOSTIC TOOL<br>
    SUISENSE v1.0 · ROBERTA-BASE · 100K TRAINING SAMPLES
</div>""", unsafe_allow_html=True)
