"""
RadVision AI - Radiology Image Analysis Platform
================================================
Requirements: pip install streamlit torch torchvision pillow google-genai numpy
Run: streamlit run app.py

Set your Gemini key via environment variable:
    export GEMINI_API_KEY="AIza..."
    streamlit run app.py
"""

import os
from dotenv import load_dotenv
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from google import genai

# Load environment variables from .env file
load_dotenv()

st.set_page_config(
    page_title="RadVision AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:       #050810;
    --bg-card:  #0b0f1e;
    --bg-card2: #0f1526;
    --cyan:     #00d4ff;
    --green:    #00e887;
    --red:      #ff3d57;
    --yellow:   #ffcc00;
    --purple:   #7c6dfa;
    --muted:    #4b5675;
    --border:   rgba(0,212,255,0.12);
    --text:     #dce8f5;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg) !important;
    color: var(--text);
}

.main, .block-container { background: var(--bg) !important; }
.block-container { padding: 2.5rem 2.5rem 2rem !important; max-width: 1400px !important; }

/* ── Top Bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 1.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(100deg, var(--cyan) 0%, var(--green) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.logo-sub {
    -webkit-text-fill-color: var(--muted);
    font-weight: 400;
    font-size: 0.82rem;
    margin-left: 12px;
    font-family: 'Inter', sans-serif;
}
.badge-live {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,232,135,0.08); border: 1px solid rgba(0,232,135,0.22);
    border-radius: 50px; padding: 5px 14px;
    font-size: 0.7rem; font-weight: 600; color: var(--green); letter-spacing: 0.07em;
}
.badge-live::before {
    content:''; width:6px; height:6px; border-radius:50%;
    background:var(--green); animation: blink 1.8s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.15} }

/* ── Panel ── */
.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 700;
    text-align: center;
    letter-spacing: 0.05em; text-transform: uppercase;
    color: var(--text); margin-bottom: 1.6rem;
    display: flex; align-items: center; gap: 10px;
}
.panel-title::after { content:''; flex:1; height:1px; background:var(--border); }

/* ── Stats ── */
.stats-row { display:flex; gap:10px; margin-bottom:1.8rem; }
.stat-chip {
    flex:1; background:var(--bg-card2);
    border:1px solid var(--border); border-radius:10px;
    padding:0.8rem 1rem; text-align:center;
}
.stat-chip .sv { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; color:var(--cyan); }
.stat-chip .sl { font-size:0.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-top:3px; }

/* ── Upload ── */
.upload-hint {
    text-align:center; padding:3rem 1rem;
    color:var(--muted); font-size:0.85rem; line-height:2;
}
.upload-hint .icon { font-size:2.8rem; margin-bottom:0.4rem; display:block; }

/* ── Detection Result ── */
.detection-result {
    border-radius:12px; padding:1.1rem 1.3rem;
    margin:1.2rem 0; display:flex; align-items:center; gap:14px;
}
.detection-result .d-icon { font-size:2rem; }
.detection-result .d-label {
    font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700;
}
.detection-result .d-sub { font-size:0.78rem; color:var(--muted); margin-top:3px; }
.result-normal   { background:rgba(0,232,135,0.07);  border:1px solid rgba(0,232,135,0.28); }
.result-cancer   { background:rgba(255,61,87,0.07);   border:1px solid rgba(255,61,87,0.28); }
.result-fracture { background:rgba(255,204,0,0.07);   border:1px solid rgba(255,204,0,0.28); }
.result-covid    { background:rgba(0,212,255,0.07);   border:1px solid rgba(0,212,255,0.28); }
.cn { color:var(--green); } .cc { color:var(--red); }
.cf { color:var(--yellow); } .cv { color:var(--cyan); }

/* ── Prob Bars ── */
.prob-wrap { margin-top:1.2rem; }
.prob-row  { display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.prob-name { font-size:0.72rem; font-weight:500; width:76px; color:var(--muted); flex-shrink:0; }
.prob-track{ flex:1; height:4px; background:rgba(255,255,255,0.05); border-radius:2px; }
.prob-fill { height:4px; border-radius:2px; }
.prob-pct  { font-size:0.72rem; width:44px; text-align:right; color:var(--text); flex-shrink:0; }

/* ── Clinical Note ── */
.clinical-note {
    background:rgba(124,109,250,0.07);
    border:1px solid rgba(124,109,250,0.18);
    border-radius:10px; padding:1rem 1.2rem;
    font-size:0.82rem; line-height:1.75; color:#a0aec0; margin-top:1.2rem;
}
.clinical-note strong { color:var(--purple); }

/* ── Chat ── */
.chat-bubble { border-radius:10px; padding:0.9rem 1.1rem; margin-bottom:0.9rem; font-size:0.86rem; line-height:1.7; }
.bubble-user { background:rgba(0,212,255,0.07); border-left:3px solid var(--cyan); }
.bubble-ai   { background:rgba(0,232,135,0.05); border-left:3px solid var(--green); }
.bubble-role { font-size:0.6rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:5px; }
.role-user { color:var(--cyan); } .role-ai { color:var(--green); }

.suggest-label { font-size:0.7rem; color:var(--muted); margin-bottom:0.7rem; }

.no-key-hint {
    text-align:center; padding:3rem 1rem;
    color:var(--muted); font-size:0.85rem; line-height:2;
}
.no-key-hint .icon { font-size:2.4rem; margin-bottom:0.5rem; display:block; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background:#07090f !important;
    border-right:1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding:1.8rem 1.2rem !important; }
.sb-label {
    font-size:0.6rem; font-weight:700; letter-spacing:0.16em;
    text-transform:uppercase; color:var(--muted); margin-bottom:0.7rem; margin-top:1.2rem;
}
.sb-chip {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:8px; padding:0.6rem 0.9rem; margin-bottom:0.45rem;
    font-size:0.78rem; color:var(--text); display:flex; justify-content:space-between;
}
.sb-chip .sv { color:var(--cyan); font-weight:600; }

/* ── Streamlit overrides ── */
div[data-testid="stFileUploader"] {
    background:var(--bg-card2) !important;
    border:1px dashed rgba(0,212,255,0.18) !important;
    border-radius:10px !important;
}
.stButton > button {
    background:linear-gradient(135deg,#00d4ff 0%,#00b3d4 100%) !important;
    color:#050810 !important; border:none !important; border-radius:8px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:0.82rem !important; letter-spacing:0.05em !important;
    width:100% !important; padding:0.7rem 1rem !important; transition:all 0.2s !important;
}
.stButton > button:hover { transform:translateY(-1px) !important; box-shadow:0 6px 24px rgba(0,212,255,0.22) !important; }
div[data-testid="stChatInput"] > div {
    background:var(--bg-card2) !important;
    border:1px solid var(--border) !important; border-radius:10px !important;
}
div[data-testid="stChatInput"] textarea { color:var(--text) !important; }
div[data-testid="stImage"] img { border-radius:10px; }
hr { border-color:var(--border) !important; margin:1.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ───────────────────────────────────────────────────────────────────
CLASSES     = ["COVID-19", "Chest Cancer", "Fracture", "Normal"]
BAR_COLOR   = {"Normal":"#00e887","Chest Cancer":"#ff3d57","Fracture":"#ffcc00","COVID-19":"#00d4ff"}
CSS_MAP     = {"Normal":"result-normal cn","Chest Cancer":"result-cancer cc","Fracture":"result-fracture cf","COVID-19":"result-covid cv"}
ICON_MAP    = {"Normal":"✅","Chest Cancer":"⚠️","Fracture":"🦴","COVID-19":"🫁"}
NOTE_MAP    = {
    "Normal":   "No significant pathological findings detected. Routine follow-up as clinically indicated.",
    "Chest Cancer":   "Potential malignant lesion identified. Immediate correlation with clinical history and tissue biopsy is recommended. Urgent oncology referral advised.",
    "Fracture": "Osseous discontinuity pattern detected. Orthopedic consultation and additional orthogonal views (AP/lateral) are advised.",
    "COVID-19": "Bilateral ground-glass opacities consistent with viral pneumonitis. RT-PCR confirmation and pulmonology review strongly recommended."
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

GEMINI_KEY = os.environ.get("GEMINI_API_KEY","")


# ─── Model ───────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(2048,256),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,len(CLASSES))
    )
    wp = "outputs/best_model.pt"
    loaded = False
    if os.path.exists(wp):
        model.load_state_dict(torch.load(wp, map_location=device))
        loaded = True
    model.to(device).eval()
    return model, device, loaded

def predict(model, device, img):
    t = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(t), dim=1)[0].cpu().tolist()
    return sorted(
        [{"class":CLASSES[i],"pct":p*100} for i,p in enumerate(probs)],
        key=lambda x: x["pct"], reverse=True
    )

def gemini_chat(key, messages, ctx=""):
    """Call Gemini API with improved error handling."""
    if not key:
        raise ValueError("❌ GEMINI_API_KEY not set. Please set environment variable: export GEMINI_API_KEY='your_key'")
    
    try:
        client = genai.Client(api_key=key)
        history = "".join(
            f"{'Radiologist' if m['role']=='user' else 'Assistant'}: {m['content']}\n"
            for m in messages[:-1]
        )
        system = (
            "You are RadVision AI — a concise, clinically precise radiology assistant. "
            "Interpret findings, suggest differentials, recommend next steps. "
            "Always advise physician review. Use **bold** for key terms. " + ctx
        )
        prompt = f"{system}\n\n{history}Radiologist: {messages[-1]['content']}\nAssistant:"
        
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        
        if not response or not response.text:
            raise ValueError("Empty response from Gemini API")
        
        return response.text
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            raise ValueError("❌ Invalid GEMINI_API_KEY. Please check your API key.")
        elif "quota" in error_msg.lower() or "rate_limit" in error_msg.lower():
            raise ValueError("⚠️ API quota exceeded. Please try again later.")
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            raise ValueError("❌ Gemini model not available. Using alternative model...")
        else:
            raise ValueError(f"Gemini API Error: {error_msg}")


# ─── Boot ────────────────────────────────────────────────────────────────────────
model, device, weights_loaded = load_model()

# ─── Top Bar ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div>
        <span class="logo">RadVision AI</span>
        <span class="logo-sub">Radiology Analysis Platform</span>
    </div>
    <div style="display:flex;align-items:center;gap:14px;">
        <span class="badge-live">SYSTEM ACTIVE</span>
        <span style="font-size:0.7rem;color:var(--muted);">
            ResNet-50 &nbsp;·&nbsp; {"Trained" if weights_loaded else "Base"} Weights &nbsp;·&nbsp; {device.upper()}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-label">Model Status</div>', unsafe_allow_html=True)
    if weights_loaded:
        st.success("✓ Trained weights loaded  (97.88% val acc)")
    else:
        st.warning("⚠ Base weights — run trainer.py first")

    st.markdown('<div class="sb-label">System Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="sb-chip"><span>Architecture</span><span class="sv">ResNet-50</span></div>
<div class="sb-chip"><span>Device</span><span class="sv">{device.upper()}</span></div>
<div class="sb-chip"><span>Input Size</span><span class="sv">224 × 224</span></div>
<div class="sb-chip"><span>Pathologies</span><span class="sv">4 Classes</span></div>
<div class="sb-chip"><span>Training Images</span><span class="sv">5,925</span></div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.65rem;color:var(--muted);line-height:1.9;padding-top:0.3rem;">⚠ For research use only.<br>Not a validated medical device.<br>Always consult a qualified radiologist.</div>', unsafe_allow_html=True)


# ─── Main Layout ─────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1,1], gap="large")

# ══ LEFT ══════════════════════════════════════════════════════════════════════════
with col_l:
    st.markdown('<div class="panel-title">🔬 Scan Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="stats-row">
    <div class="stat-chip"><div class="sv">97.9%</div><div class="sl">Val Accuracy</div></div>
    <div class="stat-chip"><div class="sv">5,925</div><div class="sl">Training Images</div></div>
    <div class="stat-chip"><div class="sv">4</div><div class="sl">Pathologies</div></div>
</div>
""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop a radiology scan", type=["png","jpg","jpeg"], label_visibility="visible")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption=uploaded.name, width="stretch")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡  Run AI Detection"):
            with st.spinner("Analyzing with ResNet-50…"):
                st.session_state["results"] = predict(model, device, img)
    else:
        st.markdown("""
<div class="upload-hint">
    <span class="icon">🩻</span>
    Upload a chest X-ray or radiology scan<br>
    <small style="font-size:0.78rem;">Supports PNG · JPG · JPEG</small>
</div>
""", unsafe_allow_html=True)

    if "results" in st.session_state:
        res = st.session_state["results"]
        top = res[0]
        css = CSS_MAP[top["class"]]

        st.markdown(f"""
<div class="detection-result {css.split()[0]}">
    <span class="d-icon">{ICON_MAP[top['class']]}</span>
    <div>
        <div class="d-label {css.split()[1]}">{top['class'].upper()}</div>
        <div class="d-sub">Primary Detection · {top['pct']:.1f}% confidence</div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="prob-wrap">', unsafe_allow_html=True)
        for r in res:
            c = BAR_COLOR[r["class"]]
            st.markdown(f"""
<div class="prob-row">
    <span class="prob-name">{r['class']}</span>
    <div class="prob-track"><div class="prob-fill" style="width:{r['pct']:.1f}%;background:{c};"></div></div>
    <span class="prob-pct">{r['pct']:.1f}%</span>
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
<div class="clinical-note">
    <strong>AI Clinical Note</strong> — {NOTE_MAP[top['class']]}
</div>
""", unsafe_allow_html=True)

# ══ RIGHT ═════════════════════════════════════════════════════════════════════════
with col_r:
    gemini_ok = bool(GEMINI_KEY)
    dot = "🟢" if gemini_ok else "🔴"
    st.markdown(f'<div class="panel-title">🤖 Clinical AI Discussion &nbsp;{dot}</div>', unsafe_allow_html=True)

    if not gemini_ok:
        st.markdown("""
<div class="no-key-hint">
    <span class="icon">💬</span>
    Gemini AI is not configured.<br>
    <small style="font-size:0.78rem;">
        Set <code>GEMINI_API_KEY</code> environment variable<br>
        and restart the app to enable clinical discussion.
    </small>
</div>
""", unsafe_allow_html=True)
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble bubble-user"><div class="bubble-role role-user">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble bubble-ai"><div class="bubble-role role-ai">RadVision AI</div>{msg["content"]}</div>', unsafe_allow_html=True)

        # Suggested prompts for AI discussion
        if "results" in st.session_state:
            top = st.session_state["results"][0]
            st.markdown('<div class="suggest-label">💡 Suggested Questions:</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"What does {top['class']} mean?", key="q1", use_container_width=True):
                    st.session_state.messages.append({"role":"user","content":f"What does {top['class']} mean?"})
                    ctx = f"Scan results: {', '.join(f'{x['class']} {x['pct']:.1f}%' for x in st.session_state['results'])}. Primary: {top['class']}. "
                    with st.spinner("Thinking…"):
                        try:
                            reply = gemini_chat(GEMINI_KEY, st.session_state.messages, ctx)
                            st.session_state.messages.append({"role":"assistant","content":reply})
                            st.rerun()
                        except Exception as e:
                            st.session_state.messages.pop()
                            st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.button("Recommended next steps?", key="q2", use_container_width=True):
                    st.session_state.messages.append({"role":"user","content":"What are the recommended next steps?"})
                    ctx = f"Scan results: {', '.join(f'{x['class']} {x['pct']:.1f}%' for x in st.session_state['results'])}. Primary: {top['class']}. "
                    with st.spinner("Thinking…"):
                        try:
                            reply = gemini_chat(GEMINI_KEY, st.session_state.messages, ctx)
                            st.session_state.messages.append({"role":"assistant","content":reply})
                            st.rerun()
                        except Exception as e:
                            st.session_state.messages.pop()
                            st.error(f"❌ Error: {str(e)}")
            
            col3, col4 = st.columns(2)
            
            with col3:
                if st.button("Treatment options?", key="q3", use_container_width=True):
                    st.session_state.messages.append({"role":"user","content":"What are the available treatment options?"})
                    ctx = f"Scan results: {', '.join(f'{x['class']} {x['pct']:.1f}%' for x in st.session_state['results'])}. Primary: {top['class']}. "
                    with st.spinner("Thinking…"):
                        try:
                            reply = gemini_chat(GEMINI_KEY, st.session_state.messages, ctx)
                            st.session_state.messages.append({"role":"assistant","content":reply})
                            st.rerun()
                        except Exception as e:
                            st.session_state.messages.pop()
                            st.error(f"❌ Error: {str(e)}")
            
            with col4:
                if st.button("Risk factors?", key="q4", use_container_width=True):
                    st.session_state.messages.append({"role":"user","content":"What are the risk factors associated with this condition?"})
                    ctx = f"Scan results: {', '.join(f'{x['class']} {x['pct']:.1f}%' for x in st.session_state['results'])}. Primary: {top['class']}. "
                    with st.spinner("Thinking…"):
                        try:
                            reply = gemini_chat(GEMINI_KEY, st.session_state.messages, ctx)
                            st.session_state.messages.append({"role":"assistant","content":reply})
                            st.rerun()
                        except Exception as e:
                            st.session_state.messages.pop()
                            st.error(f"❌ Error: {str(e)}")
            
            st.markdown("<br>", unsafe_allow_html=True)

        if prompt := st.chat_input("Ask about findings, treatment, prognosis…"):
            st.session_state.messages.append({"role":"user","content":prompt})
            ctx = ""
            if "results" in st.session_state:
                findings = ", ".join(f"{x['class']} {x['pct']:.1f}%" for x in st.session_state["results"])
                top = st.session_state["results"][0]
                ctx = f"Scan results: {findings}. Primary: {top['class']} at {top['pct']:.1f}% confidence. "
            
            with st.spinner("Thinking…"):
                try:
                    reply = gemini_chat(GEMINI_KEY, st.session_state.messages, ctx)
                    st.session_state.messages.append({"role":"assistant","content":reply})
                    st.rerun()
                except ValueError as ve:
                    # Remove the user message since there was an error
                    st.session_state.messages.pop()
                    st.error(str(ve))
                except Exception as e:
                    # Remove the user message since there was an error
                    st.session_state.messages.pop()
                    st.error(f"❌ Unexpected error: {str(e)}\n\nTroubleshooting:\n1. Check GEMINI_API_KEY is set: `echo $GEMINI_API_KEY`\n2. Verify API key is valid at: https://ai.google.dev/\n3. Restart the app: `streamlit run app.py`")

        if st.session_state.get("messages"):
            if st.button("🗑  Clear conversation"):
                st.session_state.messages = []
                st.rerun()